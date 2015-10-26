import sys, os
from glob import glob
import fitsio
import cPickle as pickle
import numpy as np
import numpy.random as npr
from CelestePy.util.misc import check_grad
from CelestePy.util.infer.optimizers import *
from scipy.optimize import minimize
sys.path.append('../../')
import redshift_utils   as ru
import GPy


###
### Experiment Params
###
SPLIT_TYPE        = "redshift"  #split_types = ["random", "flux", "redshift"]
NUM_TRAIN_EXAMPLE = 5000
NUM_BASES         = 4
BETA_VARIANCE     = 1.
BETA_LENGTHSCALE  = 40.
BASIS_DIR         = "cache/basis_locked/"

# set up experiment list
import itertools
Ks         = [3, 6, 8, 16, 32]
SPLITS     = ["random", "flux", "redshift"]
EXP_PARAMS = list(itertools.product(Ks, SPLITS))
for i,ep in enumerate(EXP_PARAMS):
    print "task_no %d: "%i, ep


if __name__=="__main__":

    ##########################################################################
    ## set sampling parameters
    ##########################################################################
    narg              = len(sys.argv)
    NUM_BASES  = 4
    SPLIT_TYPE = "random"

    print \
"""
==============================================================================
===  preproc fit basis with params
==============================================================================
    SPLIT_TYPE        = {split}
    NUM_TRAIN_EXAMPLE = {ntrain}
    NUM_BASES         = {num_bases}
    BETA_VARIANCE     = {beta_var}
    BETA_LENGTHSCALE  = {beta_ell}
    BASIS_DIR         = {bdir}
""".format(split = SPLIT_TYPE, ntrain=NUM_TRAIN_EXAMPLE,
           num_bases = NUM_BASES, beta_var = BETA_VARIANCE, beta_ell = BETA_LENGTHSCALE,
           bdir = BASIS_DIR)

    #########################################################
    #  DR10 qso dataset and spec files (train/test splits)  #
    #########################################################
    qso_psf_flux, qso_psf_flux_ivar, qso_psf_mags, qso_z, spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = SPLIT_TYPE)

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
    test_dict  = setup_data(spec_files_train[NUM_TRAIN_EXAMPLE:(NUM_TRAIN_EXAMPLE+100)])

    ########################################################################
    # Train models
    ########################################################################
    # generate som reandom data
    import matplotlib.pyplot as plt
    import seaborn as sns
    Ks = [4]
    fits = {}
    for K in Ks:
        th_train, th_test, train_lls, test_lls = \
            fit_nmf(train_dict, test_dict, K = K, num_bins=800, max_iter=50)
        fits[K] = {'th_train'  : th_train, 
                   'th_test'   : th_test,
                   'train_lls' : train_lls,
                   'test_lls'  : test_lls }
        import cPickle as pickle
        pickle.dump(fits, open('qso_basis_K_%d_split_%s.pkl'%(K, SPLIT_TYPE), 'wb'))



