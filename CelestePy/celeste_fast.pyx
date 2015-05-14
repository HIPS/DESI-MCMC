# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -std=c++11 -DEIGEN_NO_MALLOC
## distutils: extra_link_args = -fopenmp
# cython: boundscheck = False

###############################################################################
## Cythonized functions for celeste graphical model
## Author: Andrew Miller <acm@seas.harvard.edu>
###############################################################################
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp, lgamma, sqrt
from libc.stdlib cimport malloc, free
from cython cimport view
np.import_array()

# TYPEDEFS
FLOAT = np.float
INT   = np.int
ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t
ctypedef np.ulong_t INDEX_t
ctypedef np.uint8_t BOOL_t

@cython.boundscheck(False)
@cython.wraparound(False)
def gen_galaxy_psf_mixture_params(
        np.ndarray[FLOAT_t, ndim=1] thetas,
        np.ndarray[FLOAT_t, ndim=2] W,
        np.ndarray[FLOAT_t, ndim=1] v_s,
        np.ndarray[FLOAT_t, ndim=1] image_ws,
        np.ndarray[FLOAT_t, ndim=2] image_means,
        np.ndarray[FLOAT_t, ndim=3] image_covars,
        np.ndarray[FLOAT_t, ndim=1] gal_exp_amp,
        np.ndarray[FLOAT_t, ndim=1] gal_exp_sigs,
        np.ndarray[FLOAT_t, ndim=1] gal_dev_amp,
        np.ndarray[FLOAT_t, ndim=1] gal_dev_sigs
    ):
    # sanity check
    #if mus.shape[0] != sigs.shape[0] or mus.shape[0] != ws.shape[0]:
    #    raise ValueError("Means, covariances and weights must have same first dimension!")
    #if mus.shape[1] != sigs.shape[1] or mus.shape[1] != sigs.shape[2]: 
    #    raise ValueError("Means and inverse covariance shapes don't jive!")

    # iterate through and populate array of values
    cdef INDEX_t K_psf = image_ws.shape[0]
    cdef INDEX_t K_exp = gal_exp_amp.shape[0]
    cdef INDEX_t K_dev = gal_dev_amp.shape[0]

    # cache galaxy amplitudes and variances
    cdef FLOAT_t amp_ij, var_ij

    # compute MOG components
    #thetas = [theta_s, 1. - theta_s]
    num_components = image_ws.shape[0] * (gal_exp_amp.shape[0] + gal_dev_amp.shape[0])
    weights = np.zeros(num_components, dtype=np.float)
    means   = np.zeros((num_components, 2), dtype=np.float)
    covars  = np.zeros((num_components, 2, 2), dtype=np.float)
    cnt     = 0
    for k in range(K_psf):                              # num PSF Componenets
        for i in range(2):                              # two galaxy types
            Ki = K_exp if i==0 else K_dev
            #if i == 0
            #    Ki = K_exp
            #else:
            #    Ki = K_dev
            for j in range(Ki):                         # galaxy type components

                ## cache the appropriate mixture type
                if i == 0:
                    amp_ij = gal_exp_amp[j]
                    var_ij = gal_exp_sigs[j]
                else:
                    amp_ij = gal_dev_amp[j]
                    var_ij = gal_dev_sigs[j]

                ## compute weights and component mean/variances
                weights[cnt] = image_ws[k] * thetas[i] * amp_ij

                ## compute means
                means[cnt,0] = v_s[0] + image_means[k, 0]
                means[cnt,1] = v_s[1] + image_means[k, 1]

                ## compute covariance matrices
                for ii in range(2):
                    for jj in range(2):
                        covars[cnt, ii, jj] = image_covars[k, ii, jj] + \
                                              var_ij * W[ii, jj]

                # increment index
                cnt += 1
    return weights, means, covars


## generalization of the above, just does a mixture profile
@cython.boundscheck(False)
@cython.wraparound(False)
def gen_galaxy_prof_psf_mixture_params(
        np.ndarray[FLOAT_t, ndim=2] W,
        np.ndarray[FLOAT_t, ndim=1] v_s,
        np.ndarray[FLOAT_t, ndim=1] image_ws,
        np.ndarray[FLOAT_t, ndim=2] image_means,
        np.ndarray[FLOAT_t, ndim=3] image_covars,
        np.ndarray[FLOAT_t, ndim=1] gal_prof_amp,
        np.ndarray[FLOAT_t, ndim=1] gal_prof_sigs,
    ):

    # iterate through and populate array of values
    cdef INDEX_t K_psf  = image_ws.shape[0]
    cdef INDEX_t K_prof = gal_prof_amp.shape[0]

    # cache galaxy amplitudes and variances
    cdef FLOAT_t amp_ij, var_ij

    # compute MOG components
    cdef INDEX_t num_components = K_psf * K_prof
    weights = np.zeros(num_components, dtype=np.float)
    means   = np.zeros((num_components, 2), dtype=np.float)
    covars  = np.zeros((num_components, 2, 2), dtype=np.float)
    cnt     = 0
    for k in range(K_psf):                              # num PSF Componenets
        for j in range(K_prof):                         # galaxy type components
            ## compute weights and component mean/variances
            weights[cnt] = image_ws[k] * gal_prof_amp[j]

            ## compute means
            means[cnt,0] = v_s[0] + image_means[k, 0]
            means[cnt,1] = v_s[1] + image_means[k, 1]

            ## compute covariance matrices
            for ii in range(2):
                for jj in range(2):
                    covars[cnt, ii, jj] = image_covars[k, ii, jj] + \
                                          gal_prof_sigs[j] * W[ii, jj]

            # increment index
            cnt += 1
    return weights, means, covars

