import fitsio
import numpy as np
import numpy.random as npr
from scipy.optimize import minimize
from scipy import interpolate
from funkyyak import grad, numpy_wrapper as np
from scipy.optimize import minimize
from redshift_utils import load_data_clean_split, project_to_bands
from slicesample import slicesample
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sns.set_style("white")
current_palette = sns.color_palette()
npr.seed(42)

## weighted loss function - observations have gaussian noise
def loss_fun(th_mat, X, inv_var):
    N = X.shape[0]
    omegas = th_mat[:, 0:N].T
    betas  = th_mat[:, N:]
    # compute like
    W = np.exp(omegas)
    B = np.exp(betas)
    B = B / np.sum(B, axis=1, keepdims=True)
    Xtilde = np.dot(np.exp(omegas), B)
    #Xtilde = reconstruct(th_mat)
    return np.sum( inv_var * np.square(X - Xtilde) )

def reconstruct(th_mat):
    omegas = th_mat[:, 0:N].T
    betas  = th_mat[:, N:]
    # compute like
    W = np.exp(omegas)
    B = np.exp(betas)
    B = B / np.sum(B, axis=1, keepdims=True)
    Xtilde = np.dot(np.exp(omegas), B)
    return Xtilde
loss_grad = grad(loss_fun)

## joint prior over parameters
def prior_loss(th_mat, sig2_omega, N):
    omegas = th_mat[:, 0:N].T
    betas  = th_mat[:, N:]
    return 1 / (100.) * np.sum(np.square(omegas)) + \
           1 * np.sum(np.square(betas))
prior_loss_grad = grad(prior_loss)

## fixed point updates -
def loss_omegas(omegas, B, X, inv_var): 
    ll_omega = 1 / (100.) * np.sum(np.square(omegas))
    Xtilde   = np.dot(np.exp(omegas), B)
    return np.sum( inv_var * np.square(X - Xtilde) ) + ll_omega
loss_omegas_grad = grad(loss_omegas)


if __name__=="__main__":

    ## load a handful of quasar spectra
    lam_obs, qtrain, qtest = \
        load_data_clean_split(spec_fits_file = 'quasar_data.fits', 
                              Ntrain = 400)
    quasar_spectra = qtrain['spectra']
    quasar_z       = qtrain['Z']
    quasar_ivar    = qtrain['spectra_ivar']
    quasar_zerr    = qtrain['Z_err']
    N              = quasar_spectra.shape[0]

    ## load in basis
    th  = np.load("cache/basis_th.npy")
    lls = np.load("cache/lls.npy")
    lam0 = np.load("cache/lam0.npy")
    N    = th.shape[1] - lam0.shape[0]
    omegas = th[:,:N]
    betas  = th[:, N:]
    W = np.exp(omegas)
    B = np.exp(betas)
    B = B / B.sum(axis=1, keepdims=True)

    ############### pixel experiment #######################################
    def pixel_likelihood(z, w, x, lam0):
        """ compute the likelihood of 5 bands given
            z    : (scalar) red-shift of observed source
            w    : (vector) K positive weights for positive rest-frame basis
            x    : (vector) 5 pixel values corresponding to UGRIZ
            lam0 : basis wavelength values
        """
        # at rest frame for lam0
        lam_obs = lam0 * (1. + z)
        spec    = w.dot(B)
        mu      = project_to_bands(np.atleast_2d(spec), lam_obs)
        ll      = np.sum(x * np.log(mu) - mu)
        return ll

    def prior_w(w): 
        if np.any(w <= 0): 
            return -np.inf
        return 0

    ## load test example 
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 95
    if len(sys.argv) > 2:
        Nsamps = int(sys.argv[2])
    else:
        Nsamps = 5000

    print "Fitting SDSS pixel projection to test idx %d of %d (%d mcmc samps)"%(n, qtest['Z'].shape[0], Nsamps)
    spec_n             = qtest['spectra'][n, :]
    spec_n[spec_n < 0] = 0
    spec_ivar_n = qtest['spectra_ivar'][n, :]
    z_n         = qtest['Z'][n]
    mu_n        = project_to_bands(np.atleast_2d(spec_n), lam_obs)
    w_n         = W.mean(axis=1)
    x_n         = npr.poisson(mu_n).ravel()

    ## sample W's and Z's for this test example
    ll_samps = np.zeros(Nsamps)
    th_samps = np.zeros((Nsamps, len(w_n) + 1))
    th_curr  = np.concatenate((w_n, [2.5]))
    lnpdf    = lambda th: pixel_likelihood(th[-1], th[:-1], x_n, lam0) + prior_w(th[:-1])
    ll_curr  = lnpdf(th_curr)
    print "{0:15} | {1:15} | {2:15} | {3:15} ".format("iter", "log like", "z value (true z)", "weight0")
    for samp_i in range(Nsamps):
        th_curr, ll_curr = slicesample(th_curr, lnpdf, last_llh=ll_curr, 
                                       step = np.concatenate((W.shape[0]*[2], [1])),
                                       step_out = True,
                                       x_l = np.zeros(th_curr.shape), #everything is positive
                                       x_r = np.concatenate((10*W.max(axis=1), [10])),
                                       lb = -np.Inf, ub = np.Inf)
        th_samps[samp_i,:] = th_curr
        ll_samps[samp_i] = ll_curr
        if samp_i % 100 == 0:
            print "{0:15} | {1:15} | {2:15} | {3:15} ".format(
                samp_i, ll_curr, "%2.2f (%2.2f)"%(th_curr[-1], z_n), th_curr[0])
        if samp_i % 1000 == 0:
            np.save("cache/ll_samps_train_idx_%d.npy"%n, ll_samps)
            np.save("cache/th_samps_train_idx_%d.npy"%n, th_samps)
    np.save("cache/ll_samps_train_idx_%d.npy"%n, ll_samps)
    np.save("cache/th_samps_train_idx_%d.npy"%n, th_samps)


    ################# for interactive use ###################################
    if False:
        n = 144
        z_n = qtest['Z'][n]
        spec_n = qtest['spectra'][n, :]
        ll_samps = np.load("cache/ll_samps_train_idx_%d.npy"%n)
        th_samps = np.load("cache/th_samps_train_idx_%d.npy"%n)
        Nsamps = th_samps.shape[0]
    ##########################################################################

    ## reconstruct the basis from samples
    samp_idxs = np.arange(Nsamps/2, Nsamps)
    recon_samps   = th_samps[samp_idxs, :-1].dot(B)
    lam_obs_samps = np.outer(lam0, 1 + th_samps[samp_idxs, -1])
    recon_mean = np.zeros(lam_obs.shape)
    recon_samps_resampled = np.zeros((lam_obs_samps.shape[1], len(lam_obs)))
    for i in range(lam_obs_samps.shape[1]):
         recon_samps_resampled[i, :] += np.interp(lam_obs, lam_obs_samps[:, i], recon_samps[i,:])

    ## save some sample plots
    out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"
    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        out_dir = "figs/"

    fig = plt.figure(figsize=(18,6))
    plt.plot(lam_obs, spec_n, alpha = .5)
    plt.plot(lam_obs, recon_samps_resampled.mean(axis=0))
    plt.plot(lam_obs, recon_samps_resampled.mean(axis=0) + 2*recon_samps_resampled.std(axis=0), color='grey')
    plt.plot(lam_obs, recon_samps_resampled.mean(axis=0) - 2*recon_samps_resampled.std(axis=0), color='grey')
    plt.title("Quasar %d reconstruction from SDSS"%n)
    plt.xlabel("wavelength")
    plt.ylabel("$f(\lambda)$")
    plt.savefig(out_dir + "quasar_%d_mcmc_recon.pdf"%n, bbox_inches='tight')

    fig = plt.figure()
    cnts, bins, patches = plt.hist(th_samps[(Nsamps/2):, -1], 20, alpha=.5, normed=True)
    plt.xlabel("$z$ (red-shift)")
    plt.ylabel("$p(z | X, B)$")
    plt.vlines(z_n, 0,  cnts.max(), linewidth=2, color="black", label="$z_{spec}$")
    plt.vlines(th_samps[(Nsamps/2):,-1].mean(), 0, cnts.max(), linewidth=2, color='red', label="$E[z | x]$")
    plt.legend()
    plt.title("Quasar %d: red-shift posterior"%n)
    plt.xlim(th_samps[(Nsamps/2):,-1].min() - .25, th_samps[(Nsamps/2):, -1].max() + .25)
    plt.savefig(out_dir + "quasar_%d_posterior_z.pdf"%n, bbox_inches='tight')


