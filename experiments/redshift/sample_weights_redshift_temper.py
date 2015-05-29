import sys, os
import numpy as np
import quasar_infer_photometry as qip
import redshift_utils as ru
from CelestePy.util.infer.parallel_tempering import parallel_temper_slice

##############################################################################
### Start Script
##############################################################################
if __name__=="__main__":

    ##########################################################################
    ## set sampling parameters
    ##########################################################################
    narg           = len(sys.argv)
    test_n         = int(sys.argv[1]) if narg > 1 else 120 #1645
    Nsamps         = int(sys.argv[2]) if narg > 2 else 20
    Nchains        = int(sys.argv[3]) if narg > 3 else 5
    LAM_SUBSAMPLE  = int(sys.argv[4]) if narg > 4 else 10
    NUM_BASES      = int(sys.argv[5]) if narg > 5 else 4
    SPLIT_TYPE     = sys.argv[6] if narg > 6 else "redshift"  #"random", "flux", "redshift"
    SAMPLES_DIR    = sys.argv[7] if narg > 7 else "cache/photo_z_samps"
    BASIS_DIR      = sys.argv[8] if narg > 8 else "cache/basis_fits"
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
    def get_basis_sample(idx, mle = False): 
        """ Method to return a basis sample to condition on 
        (or the MLE if specified) """
        if mle: 
            return B_mle
        else:
            return B_samps[idx]

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

    ##########################################################################
    ## functions to pass into HMC
    ##########################################################################
    def ln_post(q, B):
        z     = q[0]
        omega = q[1:(B.shape[0] + 1)]
        mu    = q[B.shape[0] + 1]
        if z < 0. or z > 8.:
            return -np.inf
        ll    =  qip.pixel_likelihood(z, ru.softmax(omega), np.exp(mu), y_flux, y_flux_ivar, lam0, B)
        return ll + qip.prior_omega(omega) + qip.prior_mu(mu) + qip.prior_z(z)

    def dlnpdf(q, B):
        de = np.zeros(q.shape)
        grad_vec = np.zeros(q.shape)
        for i in range(len(q)):
            de[i] = 1e-6
            grad_vec[i] = (ln_post(q + de, B) - ln_post(q - de, B)) / 2e-6
            de[i] = 0.0
        return grad_vec

    ##########################################################################
    ## Draw samples of redshift and weights
    ##########################################################################
    temps   = np.linspace(.1, 1., Nchains)
    D       = B_mle.shape[0] + 2  # num(omegas) + m + z
    x0      = 10 * np.random.randn(len(temps), D)
    x0[:,0] = 6  * np.random.rand(len(temps))
    chain, chain_ll = parallel_temper_slice(
        lnpdf     = lambda(th): ln_post(th, B_mle),
        x0        = x0,
        Nsamps    = Nsamps,
        Nchains   = len(temps),
        temps     = temps,
        callback  = None,
        verbose   = True, 
        printskip = 50,
        compwise  = True)

    # save redshift samples
    fname = "redshift_samples_{spec}_K-{num_bases}_lamsamp-{lamsamp}_split-{split}.bin".format(
        spec      = spec_id,
        num_bases = B_mle.shape[0],
        lamsamp   = LAM_SUBSAMPLE,
        split     = SPLIT_TYPE)
    print "Saving samples to file %s"%fname
    with open(os.path.join(SAMPLES_DIR, fname), 'wb') as handle:
        np.save(handle, chain[-1])
        np.save(handle, chain_ll)

    print "z mean: %2.4f"%chain[-1, Nsamps/2:, 0].mean()
    print "z mode: %2.4f"%chain[-1, chain_ll[-1,:].argmax(), 0]
    print "z true: %2.2f"%z_n

    if False:
        fig, axarr = plt.subplots(2, 1)
        for w in range(4):
            axarr[0].plot(chain[-1,:,0])
            axarr[1].plot(chain_ll[-1,:])
        plt.show()
        plt.hist(chain[-1, Nsamps/2:, 0], 50); plt.show()

