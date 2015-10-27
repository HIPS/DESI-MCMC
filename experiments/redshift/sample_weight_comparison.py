import sys, os
import numpy as np
import cPickle as pickle
import quasar_infer_photometry as qip
import quasar_fit_basis as qfb
import redshift_utils as ru
from CelestePy.util.infer.parallel_tempering import parallel_temper_slice, slicesample
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
    SPLIT_TYPE     = sys.argv[6] if narg > 6 else "random" #"random", "flux", "redshift"
    SAMPLES_DIR    = sys.argv[7] if narg > 7 else "cache/photo_z_samps"
    BASIS_DIR      = sys.argv[8] if narg > 8 else "cache/basis_fits"
    PRIOR_TYPE     = sys.argv[9] if narg > 9 else "naive"
    NUM_TRAIN_EXAMPLE = "all"
    NUM_TEST_EXAMPLE = "all"
    SEED             = 42

    ##########################################################################
    ### load and curate basis samples
    ##########################################################################
    import cPickle as pickle
    mcem_file = "mcem_fits/qso_basis_K_6_split_%s.pkl"%SPLIT_TYPE
    model_dict = pickle.load(open(mcem_file, 'rb'))
    B_mle = model_dict[6]['Bs']
    lam0  = model_dict[6]['bins']
    lam0_delta = np.diff(lam0)

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
    if PRIOR_TYPE != "naive":
        gmm_fname = "mcem_fits/prior_weight_K_%d_split_%s.pkl"%(NUM_BASES, SPLIT_TYPE)
        with open(gmm_fname, 'rb') as handle:
            omega_dict = pickle.load(handle)
            mu_dict    = pickle.load(handle)

    ##########################################################################
    ## functions to pass into HMC
    ##########################################################################
    import simplex
    def ln_post(q, B):
        z     = q[0]
        omega = q[1:(B.shape[0])]
        w     = simplex.logit([omega]) #ru.softmax(np.concatenate([omega, [0]]))
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
        ll =  qip.pixel_likelihood(z, w, np.exp(mu), y_flux, y_flux_ivar, lam0[1:], B)
        return ll + ll_omega + ll_mu


    ##########################################################################
    ## Draw samples of redshift and weights
    ##########################################################################
    def gen_pt_chain():
        temps   = np.linspace(.1, 1., Nchains)
        D       = B_mle.shape[0] + 1  # num(omegas) + m + z
        x0      = 5 * np.random.randn(len(temps), D)
        x0[:,0] = 5 * np.random.rand(len(temps))

        def callback(s, chain, chain_lls):
            if s % 200 == 0 and s>1: 
                print " ... iteration %d, saving samples"%s
                sys.stdout.flush()
                #save_sample(s, chain, chain_lls)

        chain, chain_ll = parallel_temper_slice(
            lnpdf     = lambda(th): ln_post(th, B_mle),
            x0        = x0,
            Nsamps    = 1000,
            Nchains   = len(temps),
            temps     = temps,
            callback  = callback,
            verbose   = True, 
            printskip = 50,
            compwise  = True)
        return chain[-1, :, :]

    # gen 4 pt chains
    pt_chains = [gen_pt_chain() for _ in xrange(4)]


    ##########################################################################
    # gen ss samples
    ##########################################################################
    # NOW SLICE SAMPLE, one chain
    def gen_ss_chain():
        Nslice = 1000
        samps = np.zeros((Nslice, len(x0[0])))
        samps[0] = 5 * np.random.randn(D); samps[0,0] = 5 * np.random.rand()
        for n in range(1, Nslice):
            samps[n], ll = slicesample(samps[n-1], logprob = lambda(th): ln_post(th, B_mle))
            if n % 100 == 0:
                print "%d of %d, ll = %2.2f (z = %2.2f)"%(n, Nslice, ll, samps[n,0])
        return samps

    # gen 4 slice sample chains
    ss_chains = [gen_ss_chain() for _ in xrange(4)]


    ##########################################################################
    # plot Parallel tempering trace (traces) and slice sample traces
    ##########################################################################
    fig, axarr = plt.subplots(2, 1, figsize=(10, 5))
    for pt, ss in zip(pt_chains, ss_chains):
        axarr[0].plot(pt[:,0])
        axarr[1].plot(ss[:,0])
    axarr[1].set_xlabel("Iteration", fontsize=12)
    plt.savefig("mcmc_comparison.pdf", bbox_inches='tight')
    plt.close("all")

    # plot the resulting histograms
    fig, axarr = plt.subplots(1, 2, figsize=(10, 4))
    for pt, ss in zip(pt_chains, ss_chains):
        axarr[0].hist(pt[400:,0], bins=25, alpha=.5, normed=True)
        axarr[1].hist(ss[400:,0], bins=25, alpha=.5, normed=True)
    axarr[0].set_ylabel("$p(z)$")
    axarr[0].set_xlabel("$z$")
    axarr[1].set_xlabel("$z$")
    axarr[0].set_title("Slice within Parallel-Tempering", fontsize=14)
    axarr[1].set_title("Slice-sampling", fontsize=14)
    plt.savefig("mcmc_comparison_hist.pdf", bbox_inches='tight')
    plt.close("all")


    ##########################################################################
    # create a table that compares pt and slice ESS and Rhat
    ##########################################################################
    from CelestePy.util.infer import mcmc_diagnostics as mcd
    pt_z = np.array([pt[:,0] for pt in pt_chains])
    ss_z = np.array([ss[:,0] for ss in ss_chains])
    import pandas as pd
    table_str = pd.DataFrame({
        "PT+SS": [mcd.compute_r_hat(pt_z), mcd.compute_n_eff(pt_z), 
                    np.sum([mcd.compute_n_eff_acf(z) for z in pt_z])],
        "SS": [mcd.compute_r_hat(ss_z), mcd.compute_n_eff(ss_z), 
                    np.sum([mcd.compute_n_eff_acf(z) for z in ss_z])]
                    },
        index = ["$\hat r$", "$N_{eff}$", "ESS"]).to_latex(
            escape=False,
            float_format=lambda(th): "%2.2f"%th)

    print table_str
    with open("mcmc_comparison_table.tex", "w") as f:
        f.write(table_str)

    # gen 4 slice chains
    #print "z mean: %2.4f"%chain[-1, Nsamps/2:, 0].mean()
    #print "z mode: %2.4f"%chain[-1, chain_ll[-1,:].argmax(), 0]
    #print "z true: %2.2f"%z_n


