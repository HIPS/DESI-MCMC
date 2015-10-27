import sys, os
import numpy as np
import quasar_infer_photometry as qip
import quasar_fit_basis as qfb
import redshift_utils as ru
from sklearn import mixture
import cPickle as pickle
import numpy as np

NUM_TRAIN_EXAMPLE=2000
def load_model(split_type):
    """pretty slow way to load a fit model"""
    import sys, os; sys.path.append("experiments/")
    from qso_experiment import setup_data, setup_model

    # load master qso list
    qso_psf_flux, qso_psf_flux_ivar, qso_psf_mags, qso_z, spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = split_type)

    # randomly select a subsample to train on
    np.random.seed(0)
    spec_files_train = np.array(spec_files)[train_idx]
    np.random.shuffle(spec_files_train)
    local_specs = "/home/acm/Proj/astro/data/DR10QSO/specs/"
    spec_files_train = np.array([os.path.join(local_specs, os.path.basename(s))
                                 for s in spec_files_train])

    #######################################################################
    # load spec data!
    #######################################################################
    sys.path.append("/home/acm/Proj/astro/SEDModel/SpecExperiments/experiments/")
    from qso_experiment import setup_data, fit_nmf
    train_dict = setup_data(spec_files_train[:NUM_TRAIN_EXAMPLE])
    test_dict  = setup_data(spec_files_train[NUM_TRAIN_EXAMPLE:(NUM_TRAIN_EXAMPLE+2000)])
  
    ##############################
    # Load params if available   #
    ##############################
    import cPickle as pickle
    model_file = 'mcem_fits/qso_basis_K_%d_split_%s.pkl'%(NUM_BASES, SPLIT_TYPE)
    if os.path.exists(model_file):
        print "loading model from cache!"
        model_dict = pickle.load(open(model_file, 'rb'))
    else:
        model_dict = {NUM_BASES:None}  # otherwise init none


    ######################################
    # Combine above to create map model  #
    ######################################
    nmf_mod, _      = setup_model(test_dict, bins=model_dict['bins'], K=K)
    nmf_mod_test, _ = setup_model(val_dict, bins=model_dict['bins'], K=K)
    map_basis = nmf_mod_test.parser.get(model_dict[K]["th_test"], 'basis')

    # set map basis to each mod and test
    th_marg = np.zeros(nmf_mod.num_params())
    nmf_mod.parser.set(th_marg, 'basis', map_basis)

    th_test = np.zeros(nmf_mod_test.num_params())
    nmf_mod_test.parser.set(th_test, 'basis', map_basis)
    return nmf_mod, th_marg, nmf_mod_test, th_test



if __name__=="__main__":


    # load in trained model
    SPLIT_TYPE = "random"
    nmf_mod, th_marg, nmf_mod_test, th_test = load_model(SPLIT_TYPE)

    # collect a bunch of weight examples
    for i in range(len(test_lls)):
        th_test = nmf_valid.samp_w_marg(th_test)
        test_lls[i] = nmf_valid.loglike(th_test) / nmf_valid.N


    # normalize them, save the magnitude

    # take the logit, and fit an MOG

    # save MoG next to the original model

        # sample out of sample w's conditioned on basis
        th_test = np.zeros(nmf_valid.num_params())
        nmf_valid.parser.set(th_test, 'basis', map_basis) # (sets map basis)
        test_lls = np.zeros(200)
                #
        ll_dict[K] = test_lls
        #print np.percentile(test_lls, [1, 99])
        print np.percentile(nmf_valid.loglike_data(th_test), [2.5, 50, 97.5])









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


