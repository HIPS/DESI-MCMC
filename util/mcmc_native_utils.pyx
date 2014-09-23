cimport cython
import numpy as np
cimport numpy
from numpy cimport *
# from libc.stdlib cimport *

# The following are taken from https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
# Note that "cdef extern from" #includes the given header file; only the used parts need to be written out.

cdef extern from "randomkit.h":
  ctypedef struct rk_state:
    unsigned long key[624]
    int pos
    int has_gauss
    double gauss

cdef class RandomStateUnsafeAccess:
  # Since numpy.random.RandomState provides no means to access its pointer,
  # we assume that the state pointer is the first field after the PyObject_HEAD,
  # which is valid in all known recent versions of
  # https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx.
  # So we do what amounts to a reinterpret_cast and hope for the best!
  cdef rk_state *internal_state

cdef extern from "distributions.h":
  cdef extern long c_rk_binomial "rk_binomial" (rk_state *state, long n, double p)

ctypedef fused IMG_TYPE:
  float
  double

cdef class NativePatch:
  def __init__(self, patch):
    self.x0, self.x1, self.y0, self.y1 = patch.getExtent()
    self.data = patch.getImage()
  cdef int x0, x1, y0, y1
  cdef double [:, :] data

@cython.boundscheck(False)
def nativeSampleAuxSourceCounts(IMG_TYPE [:, :] image_data,
                                list src_patches, double eta,
                                object rand):
  
  cdef rk_state *state = (<RandomStateUnsafeAccess>rand).internal_state

  cdef int S = len(src_patches) # number of sources
  cdef int dim0 = image_data.shape[0]
  cdef int dim1 = image_data.shape[1]

  # cdef double *probs = <double *>malloc((S+1) * sizeof(double))
  cdef double [:] probs = np.zeros(S+1, dtype=np.double)
  cdef double num_probs, curr_prob
  cdef long [:] samples = np.zeros(S+1, dtype=np.int)
  cdef long curr_sample, unsampled

  cdef int i, j, s
  cdef long xij
  cdef NativePatch p
  for i in range(dim0):
    for j in range(dim1):
      sum_probs = probs[S] = eta
      for s in range(S):
        # For fast access to arrays of extension types:
        # https://groups.google.com/d/topic/cython-users/EVkjrW-vwVo
        # TODO: src_patches[s] calls __Pyx_GetItemInt_List which calls PyList_GET_ITEM,
        # so this could be faster if we made it a native array of pointers somehow.
        p = <NativePatch>src_patches[s]
        if j >= p.x0 and j < p.x1 and i >= p.y0 and i < p.y1:
          probs[s] = curr_prob = p.data[j - p.x0, i - p.y0]
          sum_probs += curr_prob
        else:
          probs[s] = curr_prob = 0

      # This allocates the image data to each of the samples[:]
      # using the conditional binomial method used by Numpy and GSL;
      # see C.S. Davis, The computer generation of multinomial random variates, 1993.
      unsampled = long(image_data[i, j])
      for s in range(S+1):
        curr_prob = probs[s]
        curr_sample = c_rk_binomial(state, unsampled, curr_prob/sum_probs)
        samples[s] += curr_sample
        unsampled -= curr_sample
        if unsampled <= 0:
          break
        sum_probs -= curr_prob
      if unsampled > 0:
        samples[S] += curr_sample

  return samples
