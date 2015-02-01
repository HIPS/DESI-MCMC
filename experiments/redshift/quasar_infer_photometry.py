import fitsio
import numpy as np
import numpy.random as npr
from scipy.optimize import minimize
from scipy import interpolate
from funkyyak import grad, numpy_wrapper as np
from redshift_utils import load_data_clean_split, project_to_bands, \
                           check_grad, softmax
from quasar_fit_basis import load_basis_fit
from hmc import hmc
import cPickle as pickle
import sys, os

############################################################################
## Likelihood and prior functions
############################################################################

### Likelihood of 5-band SDSS flux given weights, 
def pixel_likelihood(z, w, m, fluxes, fluxes_ivar, lam0, B):
    """ compute the likelihood of 5 bands given
        z    : (scalar) red-shift of observed source
        w    : (vector) K positive weights for positive rest-frame basis
        x    : (vector) 5 pixel values corresponding to UGRIZ
        lam0 : basis wavelength values
        B    : (matrix) K x P basis 
    """
    # at rest frame for lam0
    lam_obs = lam0 * (1. + z)
    spec    = np.dot(w, B)
    mu      = project_to_bands(spec, lam_obs) * m / (1. + z)
    ll      = -0.5 * np.sum(fluxes_ivar * (fluxes-mu)*(fluxes-mu))
    return ll

def prior_omega(omega):
    return -.5 * omega.dot(omega)

def dprior_omega(omega):
    return -omega

# TODO(awu): Gaussian for now
MEAN_Z_DIST = 2.5
STDEV_Z_DIST = 1.0
def prior_z(z): 
    return -(z - MEAN_Z_DIST)*(z- MEAN_Z_DIST) / (2. * STDEV_Z_DIST * STDEV_Z_DIST)
def dprior_z(z):
    return -(z - MEAN_Z_DIST) / (STDEV_Z_DIST * STDEV_Z_DIST)

# TODO(awu): lognormal for now
STDEV_M_DIST = 20.0
def prior_mu(mu):
    return - mu * mu / (2. * STDEV_M_DIST*STDEV_M_DIST)
def dprior_mu(mu):
    return - mu / (STDEV_M_DIST*STDEV_M_DIST)

##############################################################################
## Posterior sampler function - factored out for multiple chains
##############################################################################
def gen_redshift_samples(chain_idx, Nsamps, INIT_REDSHIFT, lnpdf, dlnpdf):
    """ Generate posterior samples of red-shift using HMC """
    ll_samps = np.zeros(Nsamps)

    ## initialize samples
    samps                    = np.zeros((Nsamps, B.shape[0] + 2))
    samps[0,:]               = .001 * npr.randn(B.shape[0] + 2)
    samps[0, 0]              = INIT_REDSHIFT
    samps[0, B.shape[0] + 1] = np.log(INIT_MAG)
    ll_samps[0]              = lnpdf(samps[0,:])

    ## sanity check gradient
    check_grad(fun = lnpdf, jac = dlnpdf, th = samps[0,:])

    ## sample
    Naccept         = 0
    step_sz         = .001
    avg_accept_rate = .9
    adapt_step = True
    print "{0:17}|{1:15}|{2:15}|{3:15}|{4:15}".format(
        " iter ",
        " ll ",
        " step_sz ",
        " Naccept(rate)",
        " z (z_spec)")
    for s in np.arange(1, Nsamps):
        # stop adapting after warmup
        if s > Nsamps/2:
            adapt_step = False

        # hmc draw
        samps[s,:], step_sz, avg_accept_rate = hmc(
                 U                 = lnpdf,
                 grad_U            = dlnpdf,
                 step_sz           = step_sz,
                 n_steps           = STEPS_PER_SAMPLE,
                 q_curr            = samps[s-1,:],
                 negative_log_prob = False, 
                 adaptive_step_sz  = adapt_step,
                 min_step_sz       = 0.00005,
                 avg_accept_rate   = avg_accept_rate, 
                 tgt_accept_rate   = .55)

        ## store sample
        ll_samps[s] = lnpdf(samps[s, :])
        if ll_samps[s] != ll_samps[s-1]: 
            Naccept += 1 
        if s % 5 == 0:
            print "{0:17}|{1:15}|{2:15}|{3:15}|{4:15}".format(
                "%d/%d(chain %d)"%(s, Nsamps, chain_idx),
                " %2.4f"%ll_samps[s],
                " %2.5f"%step_sz,
                " %d (%2.2f)"%(Naccept, avg_accept_rate), 
                " %2.2f (%2.2f)"%(samps[s,0], z_n))
        if s % 200:
            save_redshift_samples(samps, ll_samps, q_idx=n, chain_idx=chain_idx,
                                  K=B.shape[0], V=B.shape[1], qso_info = qso_n_info)
    ### save samples 
    save_redshift_samples(samps, ll_samps, q_idx=n, chain_idx=chain_idx,
                          K=B.shape[0], V=B.shape[1], qso_info = qso_n_info)
    return samps, ll_samps

#############################################################################
## IO functions
#############################################################################
def save_redshift_samples(th_samps, ll_samps, q_idx, K, V, qso_info, chain_idx):
    """ save basis fit info """
    #dump separately - pickle is super inefficient
    fbase = 'cache/redshift_samples_K-%d_V-%d_qso_%d_chain_%d'%(K, V, q_idx, chain_idx)
    np.save(fbase + '.npy', th_samps)
    with open(fbase + '.pkl', 'wb') as handle:
        pickle.dump(ll_samps, handle)
        pickle.dump(q_idx, handle)
        pickle.dump(qso_info, handle)
        pickle.dump(chain_idx, handle)

def load_redshift_samples(fname):
    bname = os.path.splitext(fname)[0]
    th_samples = np.load(bname + ".npy")
    with open(bname + ".pkl", 'rb') as handle:
        ll_samps = pickle.load(handle)
        q_idx    = pickle.load(handle)
        qso_info = pickle.load(handle)
        chain_idx = pickle.load(handle)
    return th_samples, ll_samps, q_idx, qso_info, chain_idx

if __name__=="__main__":

    ##########################################################################
    ## set sampling parameters
    ##########################################################################
    narg    = len(sys.argv)
    test_n  = sys.argv[1] if narg > 1 else 1
    Nsamps  = sys.argv[2] if narg > 2 else 500
    Nchains = sys.argv[3] if narg > 3 else 2

    # HMC parameters
    INIT_REDSHIFT    = 1.0
    INIT_MAG         = 10000.
    STEP_SIZE        = 0.00001
    STEPS_PER_SAMPLE = 10

    ##########################################################################
    ## Load in spectroscopically measured quasars + fluxes
    ##########################################################################
    qso_df   = fitsio.FITS('../../data/DR10QSO/DR10Q_v2.fits')[1].read()
    r_fluxes = qso_df['PSFFLUX'][:, 2]
    all_zs   = qso_df['Z_VI']

    # set TEST INDICES 
    npr.seed(13)
    test_idx = npr.permutation(len(qso_df))

    ## grab the we're sampling
    n             = test_idx[test_n]
    qso_n_info    = qso_df[n]
    z_n           = qso_n_info['Z_VI']
    y_flux        = qso_n_info['PSFFLUX']
    y_flux_ivar   = qso_n_info['IVAR_PSFFLUX']
    print "======== SAMPLING QUASAR %d ==============="%n
    print "    PLATE               = %d"%qso_n_info['PLATE']
    print "    MJD                 = %d"%qso_n_info['MJD']
    print "    FIBERID             = %d"%qso_n_info['FIBERID']
    print "    qso_idx             = %d"%n
    print "    test_idx            = %d"%test_n
    print "    z_n (percenttile)   = %2.2f (%2.2f)"%(z_n, np.sum(all_zs < z_n) / float(len(all_zs)))
    print "    r-flux (percentile) = %2.2f (%2.2f)"%(y_flux[2], np.sum(r_fluxes < y_flux[2])/float(len(r_fluxes)))

    ##########################################################################
    ## Load in basis fit (or basis samples)
    ##########################################################################
    ### load ML basis from cache (beta and omega values)
    basis_cache = 'cache/basis_fit_K-4_V-1364.pkl'
    USE_CACHE = True
    if os.path.exists(basis_cache) and USE_CACHE:
        th, lam0, lam0_delta, parser = load_basis_fit(basis_cache)

    # compute actual weights and basis values (normalized basis + weights)
    mus    = parser.get(th, 'mus')
    betas  = parser.get(th, 'betas')
    omegas = parser.get(th, 'omegas')
    W = np.exp(omegas)
    W = W / np.sum(W, axis=1, keepdims=True)
    B = np.exp(betas)
    B = B / np.sum(B * lam0_delta, axis=1, keepdims=True)
    M = np.exp(mus)

    ##########################################################################
    ## functions to pass into HMC
    ##########################################################################
    def lnpdf(q):
        z     = q[0]
        omega = q[1:(B.shape[0] + 1)]
        mu    = q[B.shape[0] + 1]
        ll    =  pixel_likelihood(z, softmax(omega), np.exp(mu), y_flux, y_flux_ivar, lam0, B)
        return ll + prior_omega(omega) + prior_mu(mu) + prior_z(z)

    def dlnpdf(q):
        de = np.zeros(q.shape)
        grad_vec = np.zeros(q.shape)
        for i in range(len(q)):
            de[i] = 1e-6
            grad_vec[i] = (lnpdf(q + de) - lnpdf(q - de)) / 2e-6
            de[i] = 0.0
        return grad_vec

    ##########################################################################
    ## Draw samples of redshift and weights
    ##########################################################################
    z_inits = [1.0, 3.0, 5.0]
    chain_samps = []
    chain_lls   = []
    for chain_idx, z_init in enumerate(z_inits):
        samps, ll_samps = gen_redshift_samples(
            chain_idx     = chain_idx,
            Nsamps        = Nsamps,
            INIT_REDSHIFT = z_init,
            lnpdf         = lnpdf,
            dlnpdf        = dlnpdf)
        chain_samps.append(samps)
        chain_lls.append(ll_samps)

    ###########################################################################
    ## debug reconstructions
    if False:
        names = np.concatenate([ ["z"], 
                                 ["w_%d"%i for i in range(B.shape[0])],
                                 ["mu"] ])
        w_samps  = np.exp(samps[:, 1:(B.shape[0]+1)])
        w_samps /= np.sum(w_samps, axis = 1, keepdims=True)
        fig, axarr = plt.subplots(2, int(np.ceil(samps.shape[1]/2.)), figsize=(8, 6))
        for i, ax in enumerate(axarr.flatten()):
            ## histogram red shift, show 
            if i >= 1 and i < B.shape[0]+1:
                pltsamps = w_samps[:, i-1]
            else:
                pltsamps = samps[Nsamps/2:, i]
 
            cnts, bins, patches = ax.hist(pltsamps, 15, normed=True, alpha=.35)
            ax.vlines(pltsamps.mean(), 0, cnts.max(), linewidth=4, 
                            color="green", label='$E[z_{photo}]$')
            if names[i] == "z":
                ax.vlines(z_n, 0, cnts.max(), linewidth=4, color="black", label='$z_{spec}$')
            ax.legend(fontsize=15)
            ax.set_title("%s"%names[i], fontsize=18)
        #plt.savefig("z_compare_idx_%d.pdf"%n, bbox_inches='tight')
        plt.show()

    ## reconstruct the basis from samples
    #samp_idxs = np.arange(Nsamps/2, Nsamps)
    #recon_samps = np.zeros((lam_obs_samps.shape[1], len(lam_obs)))
    #for i in range(len(samp_idxs)):
    #    idx = samp_idxs[i]
    #    rest_samp    = th_samps[idx, 0:-1].dot(B)
    #    lam_obs_samp = lam0 * (1 + th_samps[idx, -1])
    #    recon_samps[i, :] = np.interp(lam_obs, lam_obs_samp, rest_samp)

    ### save some sample plots
    #out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"
    #if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
    #    out_dir = "figs/"

    #fig = plt.figure(figsize=(18,6))
    #pers = np.percentile(recon_samps, [1, 50, 99], axis=0)
    #plt.plot(lam_obs, spec_n, alpha = .5)
    #plt.plot(lam_obs, pers[1])
    #plt.plot(lam_obs, pers[0], color='grey')
    #plt.plot(lam_obs, pers[2], color='grey')
    #plt.title("Quasar %d reconstruction from SDSS"%n)
    #plt.xlabel("wavelength")
    #plt.ylabel("$f(\lambda)$")
    #plt.savefig(out_dir + "quasar_%d_mcmc_recon.pdf"%n, bbox_inches='tight')

    #fig = plt.figure()
    #cnts, bins, patches = plt.hist(th_samps[(Nsamps/2):, -1], 20, alpha=.5, normed=True)
    #plt.xlabel("$z$ (red-shift)")
    #plt.ylabel("$p(z | X, B)$")
    #plt.vlines(z_n, 0,  cnts.max(), linewidth=2, color="black", label="$z_{spec}$")
    #plt.vlines(th_samps[(Nsamps/2):,-1].mean(), 0, cnts.max(), linewidth=2, color='red', label="$E[z | x]$")
    #plt.legend()
    #plt.title("Quasar %d: red-shift posterior"%n)
    #plt.xlim(th_samps[(Nsamps/2):,-1].min() - .25, th_samps[(Nsamps/2):, -1].max() + .25)
    #plt.savefig(out_dir + "quasar_%d_posterior_z.pdf"%n, bbox_inches='tight')

    ########### DEBUG ################
    if False:
        #n=95
        #z_n    = qtest['Z'][n]
        #spec_n = qtest['spectra'][n, :]
        #ivar_n = qtest['spectra_ivar'][n, :]
        Nsamps = th_samps.shape[0]
        w = fit_weights_given_basis(B, lam0, spec_n, spec_ivar_n, z_n, lam_obs)
        plt.plot(lam_obs / (1 + z_n), spec_n)
        plt.plot(lam0, w.dot(B))
        plt.plot(lam_obs / (1 + z_n), np.sqrt(spec_ivar_n), color='grey', alpha=.5)
        plt.xlim( min(lam_obs/(1+z_n)), max(lam_obs/(1+z_n)) )
        plt.show()

        # likelihood at map
        print pixel_likelihood(z_n, w, x_n, lam0)

        ## inspect w samples
        fig, axarr = plt.subplots(1, 4)
        for i, ax in enumerate(axarr.flatten()):
            cnts, bins, patches = ax.hist(th_samps[(Nsamps/2):, i], 20, alpha=.5, normed=True)
            ax.vlines(w[i], cnts.max())

    #### DEBUG ##### 
    # SANITY CHECK:  fit pixel likelihood fixed on w_n (taking the full spectrum - THIS IS CHEATING)
    #if False: 
    #    ## these guys missed: 
    #    bottom_right_misses = [ 13, 120,  94,  93,  19,  11,  89,  99, 151,  26, 166, 100,  81 ]
    #    top_left_misses = [15, 83, 14, 122, 156]
    #    n = 89
    #    z_profile = lambda z: pixel_likelihood(z, w_n, x_n, lam0, B)
    #    w_grid = np.linspace(0, 10, 100)
    #    z_grid = np.array([z_profile(z) for z in w_grid])
    #    z_grid = z_grid #- z_grid.max()
    #    fig, axarr = plt.subplots(1, 2)
    #    axarr[0].plot(w_grid, z_grid)
    #    axarr[0].vlines(z_n, z_grid[np.isfinite(z_grid)].min(), z_grid[np.isfinite(z_grid)].max())
    #    pz_grid = np.exp(z_grid - z_grid.max())
    #    axarr[1].plot(w_grid, np.exp(z_grid - z_grid.max()))
    #    axarr[1].vlines(z_n, pz_grid.min(), pz_grid.max())
    #    plt.show()


