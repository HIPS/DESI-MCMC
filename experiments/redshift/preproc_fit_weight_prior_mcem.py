import sys, os
import numpy as np
import quasar_infer_photometry as qip
import quasar_fit_basis as qfb
import redshift_utils as ru
from sklearn import mixture
import cPickle as pickle

##############################################################################
### Start Script
##############################################################################
if __name__=="__main__":

    ##########################################################################
    ## set sampling parameters
    ##########################################################################
    narg           = len(sys.argv)
    LAM_SUBSAMPLE  = int(sys.argv[4]) if narg > 4 else 10
    NUM_BASES      = int(sys.argv[5]) if narg > 5 else 4
    SPLIT_TYPE     = sys.argv[6] if narg > 6 else "redshift"  #"random", "flux", "redshift"
    BASIS_DIR      = sys.argv[9] if narg > 8 else "cache/basis_fits"
    NUM_TRAIN_EXAMPLE = 2000
    #NUM_TEST_EXAMPLE = "all"
    SEED             = 42

    ##########################################################################
    ### load and curate basis samples
    ##########################################################################
    mus, betas, omegas, th, lam0, lam0_delta, parser = \
        qip.load_fit_params(num_bases     = NUM_BASES,
                            split_type    = SPLIT_TYPE,
                            lam_subsample = LAM_SUBSAMPLE,
                            basis_dir     = BASIS_DIR)
    omegas_trans = (omegas - omegas[:,-1,np.newaxis])[:,:-1]

    ## if necessary, load more omegas that were fit to this basis

    # cross validate mog function
    def fit_mog(data, max_comps = 20):
        N            = data.shape[0]
        train        = data[:int(.75*N), :]
        test         = data[int(.75*N):, :]

        # do train/val GMM fit
        num_comps = np.arange(1, max_comps+1)
        scores    = np.zeros(len(num_comps))
        for i, num_comp in enumerate(num_comps):
            print "num_comp = %d (%d of %d)"%(num_comp, i, len(num_comps))
            g = mixture.GMM(n_components=num_comp, covariance_type='full')
            g.fit(train)
            logprobs, res = g.score_samples(test)
            scores[i] = logprobs.sum()
        print "best validation, num_comps = %d"%num_comps[scores.argmax()]

        # fit final model to all data
        g = mixture.GMM(n_components = num_comps[scores.argmax()], covariance_type='full')
        g.fit(data)
        gmm_dict = {'mean': g.means_,
                    'pis' : g.weights_,
                    'covs': g.covars_,
                    'icovs': np.array([np.linalg.inv(cov) for cov in g.covars_]),
                    'dets': np.array([np.linalg.det(cov) for cov in g.covars_]),
                    'obj': g}
        return gmm_dict

    ##########################################################################
    # Fit mixture to omegas
    # translate omegas, split train/test, fit GMM
    omega_dict = fit_mog(omegas_trans, max_comps=20)

    # fit prior to mu as well
    mu_dict = fit_mog(mus, max_comps=10)

    ##########################################################################
    # save to basis_dir
    ##########################################################################
    bfname = qfb.basis_filename(num_bases = NUM_BASES, 
                                split_type = SPLIT_TYPE,
                                lam0       = lam0)
    gmm_fname = "prior_" + bfname
    with open(os.path.join(BASIS_DIR, gmm_fname), 'wb') as handle:
        pickle.dump(omega_dict, handle)
        pickle.dump(mu_dict, handle)

    ##########################################################################
    # DEBUG/ANALYSIS
    ##########################################################################
    if False:
        # Load spec_ids from the QSO matrix file for inspection
        CACHE_TRAIN_FILE = qfb.cache_file_name(SPLIT_TYPE, NUM_TRAIN_EXAMPLE)
        if not os.path.exists(CACHE_TRAIN_FILE):
            print "Cache file not there - quitting", CACHE_TRAIN_FILE
            sys.exit(1)
        handle    = open(CACHE_TRAIN_FILE, 'rb')
        train_idx_sub  = np.load(handle)
        spec_grid      = np.load(handle)
        spec_ivar_grid = np.load(handle)
        spec_mod_grid  = np.load(handle)
        unique_lams    = np.load(handle)
        spec_zs        = np.load(handle)
        spec_ids       = np.load(handle)
        handle.close()

        import seaborn as sns
        from CelestePy.util.like.gmm_like import mog_logmarglike
        # plot low redshift qsos
        zmid = np.percentile(spec_zs, 50)
        plt.scatter(omegas[:,2], omegas[:,3], c = spec_zs.flatten())
        plt.show()

        # visualize two scatterplots, dividing into lo-z, hi-z to see if distribution
        # is invariant to z
        fig, axarr = plt.subplots(1, 2)
        spec_zs = spec_zs.flatten()
        axarr[0].scatter(omegas_trans[spec_zs < zmid,1], omegas_trans[spec_zs < zmid,2])
        axarr[1].scatter(omegas_trans[spec_zs >= zmid,1], omegas_trans[spec_zs >= zmid,2])
        plt.show()

        # histogram each marginal
        fig, axarr = plt.subplots(1, omegas_trans.shape[1])
        for i in range(len(axarr)):
            n, bins, patches = axarr[i].hist(omegas_trans[:,i], 35, normed=True)
            axarr[i].plot(bins, np.exp(mog_logmarglike(bins, means=omega_dict['mean'], covs=omega_dict['covs'], pis=omega_dict['pis'], ind=i)))
        plt.show()


        n, bins, patches = plt.hist(mus, 35, normed=True)
        plt.plot(bins, np.exp(mog_logmarglike(bins, means=mu_dict['mean'], covs = mu_dict['covs'], pis=mu_dict['pis'])))
        plt.plot(bins, np.exp(mog_logmarglike(bins, means=np.array([[mus.mean()]]), covs=np.array([[[np.cov(mus.T)]]]), pis=np.array([1]))))
        plt.show()
