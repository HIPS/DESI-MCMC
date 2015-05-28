import numpy as np
import redshift_utils   as ru
import quasar_fit_basis as qfb
import os

###
### Experiment Params
###
SPLIT_TYPE        = "flux" #redshift" #split_types = ["random", "flux", "redshift"]
NUM_TRAIN_EXAMPLE = 2000
SEED              = 42
BASIS_DIR         = "cache/basis_fits/"

if __name__=="__main__":

    print \
""" 
=============== CACHING QSO MATRICES  ====================
  split type = {split}
  num train  = {num_train}
  seed       = {seed}
  saving to  = {odir}
==========================================================
""".format(split     = SPLIT_TYPE,
           num_train = NUM_TRAIN_EXAMPLE,
           seed      = SEED,
           odir      = BASIS_DIR)

    # DR10 qso dataset and spec files
    qso_psf_flux, qso_psf_mags, qso_z, spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = SPLIT_TYPE)

    # get all train spec files
    train_spec_files = np.array(spec_files)[train_idx]

    ## randomly subselect NUM_TRAIN
    np.random.seed(SEED)
    rand_idx             = np.random.permutation(len(train_idx))
    train_idx_sub        = train_idx[rand_idx[0:NUM_TRAIN_EXAMPLE]]
    train_spec_files_sub = train_spec_files[rand_idx[0:NUM_TRAIN_EXAMPLE]]

    ## only load in NUM_TRAIN spec files
    spec_grid, spec_ivar_grid, spec_mod_grid, unique_lams, spec_zs, spec_ids, badids = \
        ru.load_specs_from_disk(train_spec_files_sub)

    # confirm we have all the same spec files
    equal = []
    for idx in xrange(len(spec_ids)):
        fname = os.path.splitext(os.path.basename(train_spec_files_sub[idx]))[0]
        equal.append(spec_ids[idx] == fname)
    print "all files are equal: ", np.all(equal)

    # keep the Visually Inspected Z's...
    spec_zs_vi = qso_z[train_idx_sub]

    ## cache 
    CACHE_TRAIN_FILE = qfb.cache_file_name(SPLIT_TYPE, NUM_TRAIN_EXAMPLE)
    with open(CACHE_TRAIN_FILE, 'wb') as handle:
        np.save(handle, train_idx)
        np.save(handle, spec_grid)
        np.save(handle, spec_ivar_grid)
        np.save(handle, spec_mod_grid)
        np.save(handle, unique_lams)
        np.save(handle, spec_zs_vi)
        np.save(handle, spec_ids)


