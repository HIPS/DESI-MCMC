import sys, os
import numpy as np
import cPickle as pickle
import quasar_infer_photometry as qip
import quasar_fit_basis as qfb
import redshift_utils as ru
from CelestePy.util.infer.parallel_tempering import parallel_temper_slice
from CelestePy.util.like.gmm_like import mog_logmarglike, mog_loglike

##############################################################################
### Start Script
##############################################################################
if __name__=="__main__":

    ##########################################################################
    ## set sampling parameters
    ##########################################################################
    print sys.argv
    narg           = len(sys.argv)
    test_n         = int(sys.argv[1]) if narg > 1 else 120 #1645
    Nsamps         = int(sys.argv[2]) if narg > 2 else 1000
    Nchains        = int(sys.argv[3]) if narg > 3 else 8
    LAM_SUBSAMPLE  = int(sys.argv[4]) if narg > 4 else 10
    NUM_BASES      = int(sys.argv[5]) if narg > 5 else 4
    SPLIT_TYPE     = sys.argv[6] if narg > 6 else "redshift"  #"random", "flux", "redshift"
    SAMPLES_DIR    = sys.argv[7] if narg > 7 else "cache/photo_z_samps"
    BASIS_DIR      = sys.argv[8] if narg > 8 else "cache/basis_fits"
    PRIOR_TYPE     = sys.argv[9] if narg > 9 else "naive"
    NUM_TRAIN_EXAMPLE = "all"
    NUM_TEST_EXAMPLE = "all"
    SEED             = 42

    ##########################################################################
    ### load and curate basis samples
    ##########################################################################
    B_mle = qip.load_basis(num_bases     = NUM_BASES,
                           split_type    = SPLIT_TYPE,
                           lam_subsample = LAM_SUBSAMPLE,
                           basis_dir     = BASIS_DIR)
    lam0, lam0_delta = ru.get_lam0(lam_subsample=LAM_SUBSAMPLE)

    ##########################################################################
    ## Load in spectroscopically measured quasars + fluxes
    ##########################################################################
    # DR10 qso dataset and spec files
    qso_psf_flux, qso_psf_flux_ivar, qso_psf_mags, qso_z, \
    spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = SPLIT_TYPE)

    ## subselect train/test to match other experiments
    if NUM_TRAIN_EXAMPLE == "all": 
        NUM_TRAIN_EXAMPLE = len(train_idx)
    if NUM_TEST_EXAMPLE == "all":
        NUM_TEST_EXAMPLE = len(test_idx)
    np.random.seed(SEED)
    rand_idx      = np.random.permutation(len(train_idx))
    train_idx_sub = train_idx[rand_idx[0:NUM_TRAIN_EXAMPLE]]
    rand_idx      = np.random.permutation(len(test_idx))
    test_idx_sub  = test_idx[rand_idx[0:NUM_TEST_EXAMPLE]]

    ## grab the we're sampling
    n             = test_idx_sub[test_n]
    spec_id       = os.path.splitext(os.path.basename(spec_files[n]))[0]
    z_n           = qso_z[n]
    y_flux        = qso_psf_flux[n, :]
    y_flux_ivar   = qso_psf_flux_ivar[n, :] #qso_n_info['IVAR_PSFFLUX']
    print \
"""
=================== SAMPLING QUASAR qso idx = {qso_idx} ================
  Quasar Info: 
    PLATE-MJD-FIBER     = {spec_id}
    qso_idx             = {qso_idx}
    test_idx            = {test_idx}
    z_n (percentile)    = {z} ({z_per})
    r-flux (percentile) = {r} ({r_per})

  MCMC Params:
    Nsamps              = {num_samps}
    Nchains             = {num_chains}

  Sampling with Model Params:
    LAM_SUBSAMPLE       = {lam_sub}
    NUM_BASES           = {num_bases}
    SPLIT_TYPE          = {split}
    BASIS_DIR           = {bdir}
    SAMPLES_DIR         = {sdir}
    SEED                = {seed}
    NUM_TEST_EXAMPLE    = {num_test}
    NUM_TRAIN_EXAMPLE   = {num_train}
========================================================================
""".format(spec_id=spec_id, qso_idx=n, test_idx=test_n,
           z         = z_n,
           z_per     = np.sum(qso_z < z_n) / float(len(qso_z)),
           r         = y_flux[2], 
           r_per     = np.sum(qso_psf_flux[:,2] < y_flux[2]) / float(len(qso_psf_flux[:,2])),
           lam_sub   = LAM_SUBSAMPLE,
           num_bases = NUM_BASES,
           split     = SPLIT_TYPE,
           bdir      = BASIS_DIR,
           sdir      = SAMPLES_DIR,
           seed      = SEED,
           num_test  = NUM_TEST_EXAMPLE,
           num_train = NUM_TRAIN_EXAMPLE,
           num_samps = Nsamps,
           num_chains = Nchains
           )
    sys.stdout.flush()

    #### load prior file
    bfname = qfb.basis_filename(num_bases = NUM_BASES, 
                                split_type = SPLIT_TYPE,
                                lam0       = lam0)
    gmm_fname = BASIS_DIR + "/prior_" + bfname
    with open(gmm_fname, 'rb') as handle:
        omega_dict = pickle.load(handle)
        mu_dict    = pickle.load(handle)

    ##########################################################################
    ## functions to pass into HMC
    ##########################################################################
    def ln_post(q, B):
        z     = q[0]
        omega = q[1:(B.shape[0])]
        w     = ru.softmax(np.concatenate([omega, [0]]))
        mu    = q[-1]
        if z < 0. or z > 8.:
            return -np.inf

        # use MOG prior?
        if PRIOR_TYPE == "naive":
            ll_omega = qip.prior_omega(omega)
            ll_mu    = qip.prior_mu(mu)
        else:
            ll_omega = mog_loglike(omega, omega_dict['mean'], omega_dict['icovs'], omega_dict['dets'], omega_dict['pis'])
            ll_mu    = mog_loglike(mu, mu_dict['mean'], mu_dict['icovs'], mu_dict['dets'], mu_dict['pis'])
        ll =  qip.pixel_likelihood(z, w, np.exp(mu), y_flux, y_flux_ivar, lam0, B)
        return ll + ll_omega + ll_mu

    def save_sample(s, chain, chain_ll):
        fname = "redshift_samples_{spec}_K-{num_bases}_lamsamp-{lamsamp}_split-{split}.bin".format(
            spec      = spec_id,
            num_bases = B_mle.shape[0],
            lamsamp   = LAM_SUBSAMPLE,
            split     = SPLIT_TYPE)
        print "   Saving samples to file %s"%os.path.join(SAMPLES_DIR, fname)
        with open(os.path.join(SAMPLES_DIR, fname), 'wb') as handle:
            np.save(handle, chain[-1,:s,:])
            np.save(handle, chain_ll[-1,:s])

    ##########################################################################
    ## Draw samples of redshift and weights
    ##########################################################################
    temps   = np.linspace(.1, 1., Nchains)
    D       = B_mle.shape[0] + 1  # num(omegas) + m + z
    x0      = 10 * np.random.randn(len(temps), D)
    x0[:,0] = 6  * np.random.rand(len(temps))

    def callback(s, chain, chain_lls):
        if s % 200 == 0 and s>1: 
            print " ... iteration %d, saving samples"%s
            sys.stdout.flush()
            save_sample(s, chain, chain_lls)

    chain, chain_ll = parallel_temper_slice(
        lnpdf     = lambda(th): ln_post(th, B_mle),
        x0        = x0,
        Nsamps    = Nsamps,
        Nchains   = len(temps),
        temps     = temps,
        callback  = callback,
        verbose   = True, 
        printskip = 50,
        compwise  = True)

    print "z mean: %2.4f"%chain[-1, Nsamps/2:, 0].mean()
    print "z mode: %2.4f"%chain[-1, chain_ll[-1,:].argmax(), 0]
    print "z true: %2.2f"%z_n

    if False:
        fig, axarr = plt.subplots(2, 1)
        axarr[0].plot(chain[-1, Nsamps/2:,:])
        axarr[1].plot(chain_ll[-1, Nsamps/2:])
        for w in range(4):
            axarr[0].plot(chain[-1,:,0])
            axarr[1].plot(chain_ll[-1,:])
        plt.show()
        plt.hist(chain[-1, Nsamps/2:, 0], 50); plt.show()



