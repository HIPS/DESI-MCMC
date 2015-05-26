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
SPLIT_TYPE        = "redshift"  #split_types = ["random", "flux", "redshift"]
NUM_TRAIN_EXAMPLE = 20000
NUM_TEST_EXAMPLE  = 10000
SEED              = 42
MIN_I             = 17.5
MAX_I             = 20.5
WIDTH_I           = .2
MAX_GAUSSIANS     = 50

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

    ### output simple prediction statistics
    test_mae = np.mean(np.abs(preds_mle - qso_z[test_idx_sub]))
    print test_mae

    ### save model output and predictions
    #outfile = "bovy_model_num_train-%d_split_type-%s.pkl"%(NUM_TRAIN_EXAMPLE, SPLIT_TYPE)
    #output = open('bovy_output.pkl', 'wb')
    #pickle.dump(model, output)
    #output.close()


