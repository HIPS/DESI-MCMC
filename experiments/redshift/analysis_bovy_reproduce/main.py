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
from bovy_gmm import bovy_xdqsoz
import redshift_utils   as ru

###
### Experiment Params
###
SPLIT_TYPE        = "random"  #split_types = ["random", "flux", "redshift"]
NUM_TRAIN_EXAMPLE = "all"       #20000
NUM_TEST_EXAMPLE  = 10000
SEED              = 42
MIN_I             = 17.5
MAX_I             = 20.5
WIDTH_I           = .2
MAX_GAUSSIANS     = 50


def save_bovy_experiment(**kwargs):
    # populate dict
    experiment_dict = {}
    for key, val in kwargs.items():
        experiment_dict[key] = val

    # create filename
    bovy_experiment_filename = \
        "bovy_exp_TRAIN-{num_train}_TEST-{num_test}_SEED-{seed}_MAXGAUSS-{maxgauss}_WIDTH-{width}.pkl".format(
            num_train = experiment_dict['NUM_TRAIN_EXAMPLE'],
            num_test  = experiment_dict['NUM_TEST_EXAMPLE'],
            seed      = experiment_dict['SEED'],
            maxgauss  = experiment_dict['MAX_GAUSSIANS'],
            width     = experiment_dict['WIDTH_I'])

    with open(bovy_experiment_filename, 'wb') as handle:
        pickle.dump(experiment_dict, handle)


if __name__=="__main__":

    print \
"""
=====================================================================
Running BOVY REPRODUCE experiment with 
  split_type = {split_type} 
  num_train  = {num_train}
  num_test   = {num_test}
  seed       = {seed}
  min_i      = {min_i}
  max_i      = {max_i}
  width_i    = {width_i}
  max_gauss  = {max_gauss}
=====================================================================
""".format(split_type = SPLIT_TYPE, 
           num_train  = NUM_TRAIN_EXAMPLE,
           num_test   = NUM_TEST_EXAMPLE,
           seed       = SEED,
           min_i      = MIN_I,
           max_i      = MAX_I, 
           width_i    = WIDTH_I,
           max_gauss  = MAX_GAUSSIANS)

    # DR10 qso dataset and spec files
    qso_psf_flux, qso_z, spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = SPLIT_TYPE)

    ## randomly subselect NUM_TRAIN
    if NUM_TRAIN_EXAMPLE == "all": 
        NUM_TRAIN_EXAMPLE = len(train_idx)
    if NUM_TEST_EXAMPLE == "all":
        NUM_TEST_EXAMPLE = len(test_idx)

    ## subselect
    np.random.seed(SEED)
    rand_idx      = np.random.permutation(len(train_idx))
    train_idx_sub = train_idx[rand_idx[0:NUM_TRAIN_EXAMPLE]]
    rand_idx      = np.random.permutation(len(test_idx))
    test_idx_sub  = test_idx[rand_idx[0:NUM_TEST_EXAMPLE]]

    # create data matrix and fit model
    data = np.column_stack((qso_psf_flux, qso_z))

    # grab 
    model, preds_mle, preds_mean = \
            bovy_xdqsoz(train_raw     = data[train_idx_sub,:],
                        test_raw      = data[test_idx_sub,:],
                        min_i         = MIN_I,
                        max_i         = MAX_I,
                        diff          = WIDTH_I,
                        max_gaussians = MAX_GAUSSIANS,
                        verbose       = True)

    # true values
    z_test   = qso_z[test_idx_sub]
    test_mae = np.mean(np.abs(preds_mle - z_test))
    print test_mae

    ## build an experiment dict for output
    save_bovy_experiment(
         experiment_name   = "bovy_gmm", 
         SPLIT_TYPE        = SPLIT_TYPE,
         NUM_TRAIN_EXAMPLE = NUM_TRAIN_EXAMPLE,
         NUM_TEST_EXAMPLE  = NUM_TEST_EXAMPLE,
         SEED              = SEED,
         MIN_I             = MIN_I,
         MAX_I             = MAX_I,
         WIDTH_I           = WIDTH_I,
         MAX_GAUSSIANS     = MAX_GAUSSIANS,
         preds_mle         = preds_mle,
         preds_mean        = preds_mean,
         z_test            = z_test,
         train_idx         = train_idx,
         train_idx_sub     = train_idx_sub,
         test_idx          = test_idx,
         test_idx_sub      = test_idx_sub
        )

