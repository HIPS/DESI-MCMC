#
# This script implements functions for maximum likelihood 
# estimation of a basis for a group of Quasar Spectra.  
#
# Roughly, this procedure is the following:
#   - Resample spectra from lam_obs into rest frame grid lam0, (using Z_spec)
#   - Fit optimize basis and weights in an NMF-like framework (with normal errors)
#
import fitsio
import numpy as np
import numpy.random as npr
from funkyyak import grad, numpy_wrapper as np
from scipy.optimize import minimize
from redshift_utils import load_data_clean_split, project_to_bands, sinc_interp, \
                           check_grad, fit_weights_given_basis, \
                           evaluate_random_direction, ParamParser, \
                           resample_rest_frame
from slicesample import slicesample
from quasar_fit_basis import load_basis_fit, make_functions
import GPy
import os
import cPickle as pickle
from elliptical_slice import elliptical_slice
from hips.inference.hmc import hmc
import seaborn as sns
sns.set_style("white")
current_palette = sns.color_palette()

def save_basis_samples(th_samps, lam0, lam0_delta, parser):
    """ save basis fit info """
    # grab B value for shape info
    B = parser.get(th_samps[0,:], 'betas')
    #dump separately - pickle is super inefficient
    np.save('cache/basis_samples_K-%d_V-%d.npy'%B.shape, th_samps)
    with open('cache/basis_samples_K-%d_V-%d.pkl'%B.shape, 'wb') as handle:
        pickle.dump(lam0, handle)
        pickle.dump(lam0_delta, handle)
        pickle.dump(parser, handle)

def load_basis_samples(fname):
    bname = os.path.splitext(fname)[0]
    th_samples = np.load(bname + ".npy")
    with open(bname + ".pkl", 'rb') as handle:
        lam0       = pickle.load(handle)
        lam0_delta = pickle.load(handle)
        parser     = pickle.load(handle)
    return th_samples, lam0, lam0_delta, parser

def gen_prior(K_chol, sig2_omega, sig2_mu):
        th = np.zeros(parser.N)
        N = parser.idxs_and_shapes['mus'][1][0]
        parser.set(th, 'betas', K_chol.dot(npr.randn(len(lam0), K)).T)
        parser.set(th, 'omegas', np.sqrt(sig2_omega) * npr.randn(N, K))
        parser.set(th, 'mus', np.sqrt(sig2_mu) * npr.randn(N))
        return th

if __name__=="__main__":

    ## load a handful of quasar spectra
    lam_obs, qtrain, qtest = \
        load_data_clean_split(spec_fits_file = 'quasar_data.fits', 
                              Ntrain = 400)

    ## construct smooth + spiky prior over betas
    beta_kern = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=40)
    K_beta = beta_kern.K(lam0.reshape((-1, 1)))
    K_chol = np.linalg.cholesky(K_beta)
    K_inv  = np.linalg.inv(K_beta)

    ## load ML basis from cache (beta and omega values)
    basis_cache = 'cache/basis_fit_K-4_V-1364.pkl'
    USE_CACHE = True
    if os.path.exists(basis_cache) and USE_CACHE:
        th_mle, lam0, lam0_delta, parser = load_basis_fit(basis_cache)
        K, V   = parser.idxs_and_shapes['betas'][1]
        Ntrain = parser.idxs_and_shapes['mus'][1][0]
        th = th_mle.copy()

    # compute actual weights and basis values (normalized basis + weights)
    mus    = parser.get(th, 'mus')
    betas  = parser.get(th, 'betas')
    omegas = parser.get(th, 'omegas')
    W_mle = np.exp(omegas)
    W_mle = W_mle / np.sum(W_mle, axis=1, keepdims=True)
    B_mle = np.exp(betas)
    B_mle = B_mle / np.sum(B_mle * lam0_delta, axis=1, keepdims=True)
    M_mle = np.exp(mus)

    ## resample to lam0 => rest frame basis 
    print "resampling de-redshifted data"
    spectra_resampled, spectra_ivar_resampled, lam_mat = \
        resample_rest_frame(qtrain['spectra'], 
                            qtrain['spectra_ivar'],
                            qtrain['Z'], 
                            lam_obs, 
                            lam0)

    ## set up the likelihood and prior functions
    ## clean nans in data and inverse variance 
    X                  = spectra_resampled
    X[np.isnan(X)]     = 0
    Lam                = spectra_ivar_resampled
    Lam[np.isnan(Lam)] = 0
    parser, loss_fun, loss_grad, prior_loss, prior_loss_grad  = \
        make_functions(X, Lam, lam0, lam0_delta, K, 
                       Kinv_beta  = K_inv,
                       K_chol     = K_chol, #np.eye(len(lam0)), # no need for chol transform here
                       sig2_omega = 1.,
                       sig2_mu    = 500.)
    print "initial loss", loss_fun(th)
    ## sanity check gradient
    check_grad(fun = lambda th: loss_fun(th) + prior_loss(th), # X, Lam), 
               jac = lambda th: loss_grad(th) + prior_loss_grad(th), #, X, Lam),
               th  = th)

    # Sample basis and weights and magnitudes for the model
    if False: #True: 
        th_samps, lam0, lam0_delta, parser = \
            load_basis_samples('cache/basis_samples_K-4_V-1364.npy')
        th = th_samps[-1, :]

    Nsamps = 200
    th_samps = np.zeros((Nsamps, len(th)))
    ll_samps = np.zeros(Nsamps)

    # initalize w/ mle
    th    = th_mle.copy()
    betas = parser.get(th, 'betas')
    betas = np.linalg.solve(K_chol, betas.T).T
    parser.set(th, 'betas', betas)

    curr_ll = -loss_fun(th) - prior_loss(th)
    print "Initial ll = ", curr_ll
    for n in range(Nsamps):
        th = hmc(U      = lambda(th): -loss_fun(th) - prior_loss(th), 
                 grad_U = lambda(th): -loss_grad(th) - prior_loss_grad(th),
                 step_sz = .0001,
                 n_steps = 10,
                 q_curr  = th,
                 negative_log_prob = False)

        ## store sample
        curr_ll = -loss_fun(th) - prior_loss(th) 
        print "%d"%n, curr_ll
        th_samps[n, :] = th
        ll_samps[n]    = curr_ll
 
    # write them out
    save_basis_samples(th_samps, lam0, lam0_delta, parser)

    ##################################################################
    # inspect posterior summary of betas
    ##################################################################
    if False: 
        # Unpack Samples
        B_samps = np.zeros((Nsamps, K, V))
        W_samps = np.zeros((Nsamps, Ntrain, K))
        M_samps = np.zeros((Nsamps, Ntrain))
        for n in range(Nsamps):
            betas = K_chol.dot(parser.get(th_samps[n, :], 'betas').T).T
            mus   = parser.get(th_samps[n, :], 'mus')
            omegas = parser.get(th_samps[n, :], 'omegas')
            W = np.exp(omegas)
            W /= np.sum(W, axis=1, keepdims=True)
            B = np.exp(betas)
            B /= np.sum(B * lam0_delta, axis=1, keepdims=True)
            M = np.exp(mus)
            B_samps[n, :] = B
            M_samps[n, :] = M.squeeze()
            W_samps[n, :] = W

        plt.plot(lam0, B_samps.mean(axis=0).T)
        plt.show()

        qidx = 20
        n, bins, patches = plt.hist(M_samps[:, qidx], normed=True)
        plt.vlines(M_mle[qidx], ymin=0, ymax=n.max())
        plt.show()

        n, bins, patches = plt.hist(W_samps[:, qidx, 1], 30, normed=True)
        plt.vlines(W_mle[qidx, 1], ymin=0, ymax=n.max(), linewidth=3 )


        def plot_recon(spec_recon, z, spec, spec_ivar):
            fig = plt.figure(figsize=(18,6))
            plt.plot(lam_obs / (1 + z), spec, linewidth=2,
                     color=current_palette[0], label="measured")
            plt.plot(lam_obs / (1 + z), spec_ivar, 
                     linewidth=1, color='grey', alpha=.5, label="inverse variance")
            plt.plot(lam0, spec_recon, linewidth=2, color=current_palette[2], label="rank %d reconstruction"%B.shape[0])
            plt.xlim(lam_obs.min()/(1 + z), lam_obs.max()/(1+z))
            plt.ylim(0, spec.max())
            plt.xlabel("wavelength")
            plt.ylabel("spectrum")
            plt.legend(fontsize='xx-large')

        qidx = 310
        samp_idx = 199
        spec_recon = M_samps[samp_idx][qidx] * \
                     W_samps[samp_idx, qidx, :].dot(B_samps[samp_idx,:,:])
        #spec_recon = np.dot(M_mle[qidx] * W_mle[qidx, :], B_mle)
        plot_recon(spec_recon, qtrain['Z'][qidx], 
                               qtrain['spectra'][qidx,:],
                               qtrain['spectra_ivar'][qidx,:])

        plt.show()


