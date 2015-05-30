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
from brescia_nn import brescia_nn
import redshift_utils   as ru

###
### Experiment Params
###
SPLIT_TYPE        = "redshift"  #split_types = ["random", "flux", "redshift"]
NUM_TRAIN_EXAMPLE = 500
NUM_TEST_EXAMPLE  = 10000
SEED              = 42
MAX_EPOCHS        = 100
NUM_LAYERS        = 4
WIDTH             = 100
FLUX_UNITS        = "nanos"

def save_brescia_experiment(**kwargs):
    # populate dict
    experiment_dict = {}
    for key, val in kwargs.items():
        experiment_dict[key] = val

    # create filename
    bovy_experiment_filename = \
        "brescia_exp_SPLIT-{split}_TRAIN-{num_train}_TEST-{num_test}_SEED-{seed}_NUM_LAYERS-{num_layers}_WIDTH-{width}_UNITS-{units}.pkl".format(
            split     = experiment_dict['SPLIT_TYPE'],
            num_train = experiment_dict['NUM_TRAIN_EXAMPLE'],
            num_test  = experiment_dict['NUM_TEST_EXAMPLE'],
            seed      = experiment_dict['SEED'],
            num_layers = experiment_dict["NUM_LAYERS"],
            width     = experiment_dict['WIDTH'],
            units     = experiment_dict['FLUX_UNITS'], 
            max_epochs = experiment_dict['MAX_EPOCHS'])

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
  width_i    = {width_i}
  num_layers = {num_layers}
  max_epochs = {max_epochs}
=====================================================================
""".format(split_type = SPLIT_TYPE, 
           num_train  = NUM_TRAIN_EXAMPLE,
           num_test   = NUM_TEST_EXAMPLE,
           seed       = SEED,
           width_i    = WIDTH,
           num_layers = NUM_LAYERS, 
           max_epochs = MAX_EPOCHS)

    # DR10 qso dataset and spec files
    qso_psf_flux, qso_psf_flux_ivar, qso_psf_flux_mags, qso_z, spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = SPLIT_TYPE)
    spec_ids = [os.path.splitext(os.path.basename(sf))[0] for sf in spec_files]

    ## randomly subselect NUM_TRAIN
    np.random.seed(SEED)
    rand_idx      = np.random.permutation(len(train_idx))
    train_idx_sub = train_idx[rand_idx[0:NUM_TRAIN_EXAMPLE]]
    rand_idx      = np.random.permutation(len(test_idx))
    test_idx_sub  = test_idx[rand_idx[0:NUM_TEST_EXAMPLE]]
    test_spec_ids = np.array(spec_ids)[test_idx_sub]

    # create data matrix and fit model
    if FLUX_UNITS == "nanos":
        data = np.column_stack((qso_psf_flux, qso_z))
    else:
        data = np.column_stack((qso_psf_mags, qso_z))

    # grab
    model, preds = \
            brescia_nn(train      = data[train_idx_sub,:],
                       test       = data[test_idx_sub,:],
                       max_epochs = MAX_EPOCHS,
                       verbose    = True)

    ### output simple prediction statistics
    z_test = qso_z[test_idx_sub]
    test_mae = np.mean(np.abs(preds - z_test))
    print test_mae

    ## build an experiment dict for output
    save_brescia_experiment(
         experiment_name   = "brescia_nn",
         model             = model,
         SPLIT_TYPE        = SPLIT_TYPE,
         NUM_TRAIN_EXAMPLE = NUM_TRAIN_EXAMPLE,
         NUM_TEST_EXAMPLE  = NUM_TEST_EXAMPLE,
         NUM_LAYERS        = NUM_LAYERS,
         MAX_EPOCHS        = MAX_EPOCHS,
         SEED              = SEED,
         WIDTH             = WIDTH,
         FLUX_UNITS        = FLUX_UNITS,
         preds             = preds,
         z_test            = z_test,
         train_idx         = train_idx,
         train_idx_sub     = train_idx_sub,
         test_idx          = test_idx,
         test_idx_sub      = test_idx_sub,
         test_spec_ids     = test_spec_ids
        )


