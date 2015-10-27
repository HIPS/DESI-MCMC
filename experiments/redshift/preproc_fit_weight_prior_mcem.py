import sys, os
import numpy as np
#import quasar_infer_photometry as qip
import quasar_fit_basis as qfb
import redshift_utils as ru
from sklearn import mixture
import cPickle as pickle
import numpy as np


##############################################################################
# Loads model for training and validation data, ensures MAP basis is correct
##############################################################################
def load_model(split_type):
    """pretty slow way to load a fit model"""
    import sys, os; sys.path.append("../../../SEDModel/SpecExperiments/experiments/")
    from qso_experiment import setup_data, setup_model, fit_nmf

    ## load master qso list
    qso_psf_flux, qso_psf_flux_ivar, qso_psf_mags, qso_z, spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = split_type)

    ## Load model fit (and num train examples #
    NUM_BASES = 6
    model_file = 'mcem_fits/qso_basis_K_%d_split_%s.pkl'%(NUM_BASES, split_type)
    if os.path.exists(model_file):
        print "loading model from cache!"
        model_dict = pickle.load(open(model_file, 'rb'))
    else:
        model_dict = {NUM_BASES:None}  # otherwise init none
    NUM_TRAIN_EXAMPLE = model_dict[NUM_BASES]['NUM_TRAIN_EXAMPLE']
    bins = model_dict[NUM_BASES]['bins']

    ## load spec data!
    np.random.seed(0)
    spec_files_train = np.array(spec_files)[train_idx]
    np.random.shuffle(spec_files_train)
    local_specs = "/home/acm/Proj/astro/data/DR10QSO/specs/"
    spec_files_train = np.array([os.path.join(local_specs, os.path.basename(s))
                                 for s in spec_files_train])

    train_dict = setup_data(spec_files_train[:NUM_TRAIN_EXAMPLE])
    test_dict  = setup_data(spec_files_train[NUM_TRAIN_EXAMPLE:(NUM_TRAIN_EXAMPLE+2000)])

    ## Combine above to create map model
    nmf_mod, _      = setup_model(train_dict, bins=bins, K=NUM_BASES)
    nmf_mod_test, _ = setup_model(test_dict,  bins=bins, K=NUM_BASES)
    map_basis = nmf_mod.parser.get(model_dict[NUM_BASES]["th_train"], 'basis')

    # set map basis to each mod and test
    th_marg = np.zeros(nmf_mod.num_params())
    nmf_mod.parser.set(th_marg, 'basis', map_basis)

    th_test = np.zeros(nmf_mod_test.num_params())
    nmf_mod_test.parser.set(th_test, 'basis', map_basis)
    return nmf_mod, th_marg, nmf_mod_test, th_test


###########################################################################
# cross validate mog function
###########################################################################
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


##############################################################################
# main fitting function
##############################################################################
def fit_weight_prior(split_type):
    # load model
    nmf_mod, th_marg, nmf_mod_test, th_test = load_model(split_type)

    # collect a bunch of weight examples
    for i in range(100):
        if i % 10 == 0: print "... %d of %d"%(i, 100)
        th_marg = nmf_mod.samp_w_marg(th_marg)
        th_test = nmf_mod_test.samp_w_marg(th_test)

    # make sure the loglikes are reasonable
    print nmf_mod.loglike(th_marg) / nmf_mod.N
    print nmf_mod_test.loglike(th_test) / nmf_mod_test.N

    # normalize them, save the magnitude
    ws_train, Bs_train = nmf_mod.vec_to_params(th_marg)
    ws_val, Bs_val     = nmf_mod_test.vec_to_params(th_test)
    assert np.allclose(Bs_train, Bs_val), "bases should be the same!"
    ws = np.row_stack((ws_train, ws_val))
    ms = np.sum(ws, axis=1)
    ws = ws / ms[:,None]

    # take the logit of omegas, and fit an MOG
    import simplex
    omegas = simplex.logit(ws)

    # remove rows where omegas are nan
    rows, cols = np.where(np.isnan(omegas) | np.isinf(omegas))
    mask = np.ones(omegas.shape[0], dtype=bool)
    mask[rows] = False
    omegas = omegas[mask,:]
    omega_fit = fit_mog(omegas, max_comps=30)

    # fit MOG to log(mags) as well
    mus    = np.log(ms)
    mu_fit = fit_mog(mus.reshape((-1, 1)), max_comps=30)
    return omega_fit, mu_fit


if __name__=="__main__":

    # load in trained model
    split_types = ["random", "flux", "redshift"]
    for split_type in split_types:
        omega_fit, mu_fit = fit_weight_prior(split_type)

        ## save to basis_dir
        NUM_BASES = omega_fit['mean'].shape[1]+1
        model_file = 'mcem_fits/prior_weight_K_%d_split_%s.pkl'%(NUM_BASES, split_type)
        with open(model_file, 'wb') as f:
            pickle.dump(omega_fit, f)
            pickle.dump(mu_fit, f)

