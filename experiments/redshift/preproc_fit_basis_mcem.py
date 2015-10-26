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

###################
#  CLI interface  #
###################
import argparse
parser = argparse.ArgumentParser(description="Fit bases with MCEM")
parser.add_argument('-K', '--num_bases', help='Number of bases to fit',
                    required=True)
parser.add_argument('-N', '--num_train', help='Number of training examples to use',
                    required=True)
parser.add_argument('-S', '--split_type', help='Split type to use (random, flux, redshift)',
                    required=True)
args = vars(parser.parse_args())


###
### Experiment Params
###
SPLIT_TYPE        = args['split_type']  #split_types = ["random", "flux", "redshift"]
NUM_TRAIN_EXAMPLE = int(args['num_train'])
NUM_BASES         = int(args['num_bases'])
BETA_VARIANCE     = 1.
BETA_LENGTHSCALE  = 40.
BASIS_DIR         = "cache/basis_locked/"

# set up experiment list
if __name__=="__main__":

    ##########################################################################
    ## set sampling parameters
    ##########################################################################
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
           num_bases = NUM_BASES,
           beta_var = BETA_VARIANCE, beta_ell = BETA_LENGTHSCALE,
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


    ########################################################################
    # Train models
    ########################################################################
    # generate som reandom data
    import matplotlib.pyplot as plt
    import seaborn as sns
    fits = {}
    th_train, th_test, train_lls, test_lls, bins, ws_train, Bs = \
        fit_nmf(train_dict, test_dict, K = NUM_BASES, num_bins=800, 
                max_iter=10, th_init=model_dict[NUM_BASES]['th_train'])
    fits[NUM_BASES] = {'th_train'  : th_train,
                       'th_test'   : th_test,
                       'train_lls' : train_lls,
                       'test_lls'  : test_lls,
                       'bins'      : bins,
                       'ws_train'  : ws_train,
                       'Bs'        : Bs }
    pickle.dump(fits, open('mcem_fits/qso_basis_K_%d_split_%s.pkl'%(NUM_BASES, SPLIT_TYPE), 'wb'))



