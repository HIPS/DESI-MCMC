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
                           resample_rest_frame, get_lam0
from slicesample import slicesample
from quasar_fit_basis import load_basis_fit, make_functions
import GPy
import os, sys
import cPickle as pickle
from hmc import hmc
#import seaborn as sns
#sns.set_style("white")
#current_palette = sns.color_palette()

def save_basis_samples(th_samps, ll_samps, lam0, lam0_delta, parser, chain_idx):
    """ save basis fit info """
    # grab B value for shape info
    B = parser.get(th_samps[0,:], 'betas')
    #dump separately - pickle is super inefficient
    fbase = 'cache/basis_samples_K-%d_V-%d_chain_%d'%(B.shape[0], B.shape[1], chain_idx)
    np.save(fbase + '.npy', th_samps)
    with open(fbase + '.pkl', 'wb') as handle:
        pickle.dump(ll_samps, handle)
        pickle.dump(lam0, handle)
        pickle.dump(lam0_delta, handle)
        pickle.dump(parser, handle)
        pickle.dump(chain_idx, handle)

def load_basis_samples(fname):
    bname = os.path.splitext(fname)[0]
    th_samples = np.load(bname + ".npy")
    with open(bname + ".pkl", 'rb') as handle:
        ll_samps   = pickle.load(handle)
        lam0       = pickle.load(handle)
        lam0_delta = pickle.load(handle)
        parser     = pickle.load(handle)
        chain_idx  = pickle.load(handle)
    return th_samples, ll_samps, lam0, lam0_delta, parser, chain_idx

def gen_prior(K_chol, sig2_omega, sig2_mu):
        th = np.zeros(parser.N)
        N = parser.idxs_and_shapes['mus'][1][0]
        parser.set(th, 'betas', K_chol.dot(npr.randn(len(lam0), K)).T)
        parser.set(th, 'omegas', np.sqrt(sig2_omega) * npr.randn(N, K))
        parser.set(th, 'mus', np.sqrt(sig2_mu) * npr.randn(N))
        return th

if __name__=="__main__":

    ##################################################################
    ## SET INPUT PARAMS
    ##################################################################
    chain_idx    = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    Nsamps       = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    length_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 40.
    init_iter    = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    K            = 4
    npr.seed(chain_idx)  ## different random seed
    print "==== SAMPLING CHAIN ID = %d "%chain_idx

    ##################################################################
    ## load a handful of quasar spectra and resample
    ##################################################################
    lam_obs, qtrain, qtest = \
        load_data_clean_split(spec_fits_file = 'quasar_data.fits',
                              Ntrain = 400)
    N = qtrain['spectra'].shape[0]

    ## resample to lam0 => rest frame basis 
    lam0, lam0_delta = get_lam0(lam_subsample=10)
    print "resampling de-redshifted data"
    spectra_resampled, spectra_ivar_resampled, lam_mat = \
        resample_rest_frame(qtrain['spectra'], 
                            qtrain['spectra_ivar'],
                            qtrain['Z'], 
                            lam_obs, 
                            lam0)
    # clean nans
    X                  = spectra_resampled
    X[np.isnan(X)]     = 0
    Lam                = spectra_ivar_resampled
    Lam[np.isnan(Lam)] = 0

    ###########################################################################
    ## Set prior variables (K_chol, sig2_omega, sig2_mu) 
    ###########################################################################
    sig2_omega = 1.
    sig2_mu    = 100.
    beta_kern = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=length_scale)
    K_beta    = beta_kern.K(lam0.reshape((-1, 1)))
    K_chol    = np.linalg.cholesky(K_beta)
    K_inv     = np.linalg.inv(K_beta)

    ##########################################################################
    ## set up the likelihood and prior functions and generate a sample
    ##########################################################################
    parser, loss_fun, loss_grad, prior_loss, prior_grad  = \
        make_functions(X, Lam, lam0, lam0_delta, K, 
                       Kinv_beta  = K_inv,
                       K_chol     = K_chol,
                       sig2_omega = 1.,
                       sig2_mu    = 500.)
    # sample from prior
    th = np.zeros(parser.N)
    parser.set(th, 'betas', .01 * np.random.randn(K, len(lam0)))
    parser.set(th, 'omegas', .01 * npr.randn(N, K))
    parser.set(th, 'mus', .01 * npr.randn(N))

    #print "initial loss", loss_fun(th)
    check_grad(fun = lambda th: loss_fun(th) + prior_loss(th), # X, Lam), 
               jac = lambda th: loss_grad(th) + prior_grad(th), #, X, Lam),
               th  = th)

    ###########################################################################
    ## optimize for about 350 iterations to get to some meaty part of the dist
    ###########################################################################
    print "Switching to CVX Optimization Method"
    res = minimize(fun = lambda(th): loss_fun(th) + prior_loss(th),
                   jac = lambda(th): loss_grad(th) + prior_grad(th),
                   x0  = th, 
                   method = 'L-BFGS-B',
                   options = {'gtol':1e-6, 'ftol':1e-6, 
                              'disp':True, 'maxiter':init_iter})
    th = res.x

    ##########################################################################
    # Sample Nsamps, adapt step size
    ##########################################################################
    # Sample basis and weights and magnitudes for the model
    th_samps = np.zeros((Nsamps, len(th)))
    ll_samps = np.zeros(Nsamps)
    curr_ll  = -loss_fun(th) - prior_loss(th)
    print "Initial ll = ", curr_ll
    print "{0:15}|{1:15}|{2:15}|{3:15}|{4:15}".format(
        "  iter   ",
        "  lnpdf  ",
        "  step_size  ",
        "  accept rate ",
        "  num accepted")
    step_sz         = .0008
    avg_accept_rate = .9
    Naccept         = 0
    for n in range(Nsamps):
        th, step_sz, avg_accept_rate = hmc(
                 U      = lambda(th): -loss_fun(th) - prior_loss(th),
                 grad_U = lambda(th): -loss_grad(th) - prior_grad(th),
                 step_sz = step_sz,
                 n_steps = 20,
                 q_curr  = th,
                 negative_log_prob = False, 
                 adaptive_step_sz = True, 
                 avg_accept_rate = avg_accept_rate, 
                 tgt_accept_rate = .8)

        ## store sample
        th_ll = -loss_fun(th) - prior_loss(th)
        if th_ll != curr_ll:
            Naccept += 1 
        curr_ll = -loss_fun(th) - prior_loss(th) 
        th_samps[n, :] = th
        ll_samps[n]    = curr_ll
        if n % 10 == 0:
            print "{0:15}|{1:15}|{2:15}|{3:15}|{4:15}".format(
                " %d / %d "%(n, Nsamps),
                " %2.4f"%ll_samps[n],
                " %2.5f"%step_sz,
                " %2.3f"%avg_accept_rate,
                " %d "%Naccept)
        if n % 200 == 0: 
            save_basis_samples(th_samps, ll_samps, lam0, lam0_delta, parser, chain_idx)

    # write them out
    save_basis_samples(th_samps, ll_samps, lam0, lam0_delta, parser, chain_idx)

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

        qidx = 30
        n, bins, patches = plt.hist(M_samps[:, qidx], 25, normed=True)
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

        qidx = 20
        samp_idx = 1409
        spec_recon = M_samps[samp_idx][qidx] * \
                     W_samps[samp_idx, qidx, :].dot(B_samps[samp_idx,:,:])
        #spec_recon = np.dot(M_mle[qidx] * W_mle[qidx, :], B_mle)
        plot_recon(spec_recon, qtrain['Z'][qidx], 
                               qtrain['spectra'][qidx,:],
                               qtrain['spectra_ivar'][qidx,:])

        plt.show()


