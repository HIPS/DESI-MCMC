# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -std=c++11 -DEIGEN_NO_MALLOC -fopenmp
# distutils: extra_link_args = -fopenmp
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

###############################################################################
## Cythonized functions for celeste graphical model
## Author: Andrew Miller <acm@seas.harvard.edu>
###############################################################################
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, exp, sqrt
from cython cimport view
from cython.parallel import prange
np.import_array()

# TYPEDEFS
FLOAT = np.float
INT   = np.int
ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t
ctypedef np.ulong_t INDEX_t
ctypedef np.uint8_t BOOL_t

# store log(2*PI)
cdef FLOAT_t log2pi = 1.8378770664093453

@cython.boundscheck(False)
@cython.wraparound(False)
def multivariate_normal_2d_covinv_logdet_logpdf(
        np.ndarray[FLOAT_t, ndim=1] lls,       # buffer to place log likelihood vals
        np.ndarray[FLOAT_t, ndim=2] x,         # N x 2 matrix of 2 D points
        np.ndarray[FLOAT_t, ndim=1] mean,      # mean of normal, 
        np.ndarray[FLOAT_t, ndim=2] covinv,    # inverse covariance
        float logdet):
    """ Log density of a multivariate normal distribution 
        Input:
          - x  : N x D array of data (N iid)
          - mu : D length vector, mean values
          - cov: D x D array, covariance

          For tightening up loops:
          - logdet : log determinant of the covariance matrix
          - covinv : inverse of the covariance (for speed-ups)
    """
    # sanity check
    if mean.shape[0] != x.shape[1]:
        raise ValueError("data dimension 1 doesn't equal mean dimension 0")
    if mean.shape[0] != covinv.shape[0] or mean.shape[0] != covinv.shape[1]:
        raise ValueError("inverse covariance and mean shape aren't equal!")
    if mean.shape[0] != 2: 
        raise ValueError("must be two-dimensional!")

    # iterate through and populate array of values
    cdef INDEX_t D = mean.shape[0]
    cdef INDEX_t N = x.shape[0]
    cdef INDEX_t n, d
    cdef FLOAT_t quad_form, x_sig_0, x_sig_1

    # compute inner product:  mu^T Siginv mu
    cdef FLOAT_t mean_sig = mean[0] * mean[0] * covinv[0, 0] + \
                            mean[1] * mean[1] * covinv[1, 1] + \
                            2. * mean[0] * mean[1] * covinv[0, 1]
    for n in range(N):
        # compute quadratic, (x^T Siginv x - 2 mu^T Siginv x + mu^T Siginv mu)
        quad_form = 0.0
        x_sig_0 = x[n, 0] * covinv[0, 0] + x[n, 1] * covinv[1, 0]
        x_sig_1 = x[n, 0] * covinv[0, 1] + x[n, 1] * covinv[1, 1]
        quad_form =        (x[n, 0] * x_sig_0 + x[n, 1] * x_sig_1) \
                    - 2. * (mean[0] * x_sig_0 + mean[1] * x_sig_1) \
                    + mean_sig
        lls[n] = -.5 * D * log2pi - .5 * logdet - .5 * quad_form
    return lls

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def gmm_like_2d_covinv_logdet(
#        np.ndarray[FLOAT_t, ndim=1] probs,      # buffer to place prob values
#        np.ndarray[FLOAT_t, ndim=2] x,          # N x 2 matrix of 2 D points
#        np.ndarray[FLOAT_t, ndim=1] ws,         # mixing weights
#        np.ndarray[FLOAT_t, ndim=2] mus,        # mean of normal, 
#        np.ndarray[FLOAT_t, ndim=3] invsigs,    # inverse covariance
#        np.ndarray[FLOAT_t, ndim=1] logdets):
#    """ Gaussian Mixture Model likelihood
#        Input:
#          - x    = N x D array of data (N iid)
#          - ws   = K length vector that sums to 1, mixing weights
#          - mus  = K x D array of mixture component means
#          - sigs = K x D x D array of mixture component covariances
#
#          - invsigs = K x D x D array of mixture component covariance inverses
#          - logdets = K array of mixture component covariance logdets
#
#        Output:
#          - N length array of log likelihood values
#    """
#    # sanity check
#    if mus.shape[0] != invsigs.shape[0] or mus.shape[0] != ws.shape[0]:
#        raise ValueError("Means, covariances and weights must have same first dimension!")
#    if mus.shape[1] != invsigs.shape[1] or mus.shape[1] != invsigs.shape[2]: 
#        raise ValueError("Means and inverse covariance shapes don't jive!")
#
#    # iterate through and populate array of values
#    cdef INDEX_t K = mus.shape[0]
#    cdef INDEX_t N = x.shape[0]
#    cdef INT_t n, k
#    cdef FLOAT_t x_center_0, x_center_1, quad_form
#
#    # zero out likelihood
#    for n in range(N):
#        probs[n] = 0.0
#
#    # compute component-wise likelihoods
#    for k in range(K):
#        for n in range(N):
#            x_center_0 = x[n, 0] - mus[k, 0]
#            x_center_1 = x[n, 1] - mus[k, 1]
#            quad_form  =   x_center_0 * x_center_0 * invsigs[k, 0, 0] \
#                         + x_center_1 * x_center_1 * invsigs[k, 1, 1] \
#                         + 2. * x_center_0 * x_center_1 * invsigs[k, 0, 1]
#            probs[n] += exp(-log2pi - .5 * logdets[k] - .5 * quad_form) * ws[k]
#

@cython.boundscheck(False)
@cython.wraparound(False)
def gmm_like_2d(FLOAT_t[::1]      probs,      # buffer to place prob values
                FLOAT_t[:,::1]    x,          # N x 2 matrix of 2 D points
                FLOAT_t[::1]      ws,         # mixing weights
                FLOAT_t[:,::1]    mus,        # mean of normal, 
                FLOAT_t[:,:,::1]  sigs):      # inverse covariance
    """ Gaussian Mixture Model likelihood
        Input:
          - x    = N x D array of data (N iid)
          - ws   = K length vector that sums to 1, mixing weights
          - mus  = K x D array of mixture component means
          - sigs = K x D x D array of mixture component covariances

        Output:
          - N length array of log likelihood values
    """
    # sanity check
    if mus.shape[0] != sigs.shape[0] or mus.shape[0] != ws.shape[0]:
        raise ValueError("Means, covariances and weights must have same first dimension!")
    if mus.shape[1] != sigs.shape[1] or mus.shape[1] != sigs.shape[2]: 
        raise ValueError("Means and inverse covariance shapes don't jive!")

    # iterate through and populate array of values
    cdef INDEX_t K = mus.shape[0]
    cdef INDEX_t N = x.shape[0]
    cdef INDEX_t n, k
    cdef FLOAT_t x_center_0, x_center_1, quad_form, invk_00, invk_01, invk_11, detk

    # zero out likelihood
    for n in prange(N, nogil=True):
        probs[n] = 0.0

    # compute component-wise likelihoods
    for k in range(K):

        # compute determinant of component K's covariance matrix
        detk = sigs[k, 0, 0] * sigs[k, 1, 1] - sigs[k, 0, 1] * sigs[k, 1, 0]
        invk_00 = sigs[k, 1, 1] / detk
        invk_11 = sigs[k, 0, 0] / detk
        invk_01 = -1 * sigs[k, 0, 1] / detk

        for n in prange(N, nogil=True):
            x_center_0 = x[n, 0] - mus[k, 0]
            x_center_1 = x[n, 1] - mus[k, 1]
            quad_form  =   x_center_0 * x_center_0 * invk_00 \
                         + x_center_1 * x_center_1 * invk_11 \
                         + 2. * x_center_0 * x_center_1 * invk_01
            probs[n] += exp(-log2pi - .5 * log(detk) - .5 * quad_form) * ws[k]


