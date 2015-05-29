import sys, os
import numpy as np
import quasar_infer_photometry as qip
import redshift_utils as ru
from sklearn import mixture

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
    #NUM_TRAIN_EXAMPLE = "all"
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

    num_comp = 4
    g = mixture.GMM(n_components=num_comp, covariance_type='full')


