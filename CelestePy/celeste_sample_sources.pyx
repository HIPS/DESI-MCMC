import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

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

# Cython TypeDefs
ctypedef np.float_t FLOAT_t
ctypedef np.ulong_t INDEX_t


cdef class NativePatch:
    """Data for a small patch within a big field - keeps a pointer to a 
    typed memory view (2d) of image data, as well as where it is in the bigger
    field: lower left = (x0, y0), upper right = (x1, y1)
    """
    def __init__(self, FLOAT_t[:,::1] data, ylim, xlim):
        self.data = data
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
    cdef public int x0, x1, y0, y1
    cdef public FLOAT_t[:,::1] data


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline vector[INDEX_t] get_bounding_boxes_idx(INDEX_t x, INDEX_t y, INDEX_t[:,::1] src_boxes):
    cdef INDEX_t S = src_boxes.shape[0]
    cdef INDEX_t s

    # create list of source indices
    cdef vector[INDEX_t] vect
    for s in range(S):
        if x > src_boxes[s,0] and x < src_boxes[s,1] and \
           y > src_boxes[s,2] and y < src_boxes[s,3]:
            vect.push_back(s)
    return vect


@cython.wraparound(False)
@cython.boundscheck(False)
def sample_source_counts(list src_imgs,
                         list samp_imgs,
                         FLOAT_t[:,::1] nelec,
                         INDEX_t[:,::1] src_boxes,
                         FLOAT_t epsilon,
                         object random_state=None):

    # create ranodm state if not none
    random_state = random_state or np.random.RandomState()
    cdef rk_state *state = (<RandomStateUnsafeAccess>random_state).internal_state

    # keep track of noise sum in image (for inferring img.epsilon)
    cdef INDEX_t dim0 = nelec.shape[0]
    cdef INDEX_t dim1 = nelec.shape[1]
    cdef FLOAT_t noise_sum = 0.
    cdef INDEX_t cnt = 0
    cdef FLOAT_t num_photons_xy
    cdef INDEX_t x, y, x0, x1, y0, y1, pi, i, S
    cdef vector[INDEX_t] possible_srcs
    cdef NativePatch impatch
    cdef long [:] src_photons
    cdef FLOAT_t [::1] model_fluxes
    for y in range(dim0):
        for x in range(dim1):
            num_photons_xy = nelec[y, x]

            # get possible sources for the pixel at x, y
            possible_srcs = get_bounding_boxes_idx(x, y, src_boxes)
            if possible_srcs.size() == 0:
                noise_sum += num_photons_xy
                continue

            # if only one possible source, don't sample from multinomial...
            S = possible_srcs.size() + 1
            model_fluxes = np.zeros(S, dtype=np.float)
            for i in range(S-1):
                pi              = possible_srcs[i] #possible_srcs.front()
                impatch         = <NativePatch>src_imgs[pi]
                model_fluxes[i] = impatch.data[y-impatch.y0, x-impatch.x0]

            # tack on sky noise on the end
            model_fluxes[S-1] = epsilon

            # multinomial sample
            # p = model_fluxes / model_fluxes.sum()
            src_photons = sample_multinomial(int(num_photons_xy),
                                             model_fluxes,
                                             state)

            # store in sampled images
            for i in range(possible_srcs.size()):
                pi      = possible_srcs[i] #.front(); possible_srcs.pop()
                impatch = <NativePatch>samp_imgs[pi]
                impatch.data[y-impatch.y0, x-impatch.x0] = src_photons[i]

            # increment sky noise summary
            noise_sum += src_photons[S-1]

            # verbose??
            cnt += 1
            #if cnt % 100000 == 0:
            #    print "%d of %d"%(cnt, dim0*dim1)
    return samp_imgs, noise_sum


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline long[:] sample_multinomial(
            np.int_t     n,
            FLOAT_t[::1] probs,
            rk_state*    state):
    """sample a multinomial random variate"""
    # This allocates the image data to each of the samples[:]
    # using the conditional binomial method used by Numpy and GSL;
    # see C.S. Davis, The computer generation of multinomial random variates, 1993.
    #random_state = random_state or np.random.RandomState()
    #cdef rk_state *state = (<RandomStateUnsafeAccess>random_state).internal_state
    cdef long S = probs.shape[0]
    cdef long [:] samples = np.zeros(S, dtype=np.int)
    cdef long curr_sample, s
    cdef double num_probs, curr_prob

    cdef long unsampled = long(n)
    cdef double sum_probs = 0.
    for s in range(S):
        sum_probs += probs[s]

    for s in range(S-1):
        curr_prob = probs[s]
        curr_sample = c_rk_binomial(state, unsampled, curr_prob/sum_probs)
        samples[s] += curr_sample
        unsampled  -= curr_sample
        if unsampled <= 0:
            break
        sum_probs -= curr_prob
    samples[S-1] += unsampled
    return samples


#ctypedef fused IMG_TYPE:
#  float
#  double
#@cython.boundscheck(False)
#def nativeSampleAuxSourceCounts(IMG_TYPE [:, :] image_data,
#                                list src_patches, double eta,
#                                object rand):
#  
#  cdef rk_state *state = (<RandomStateUnsafeAccess>rand).internal_state
#
#  cdef int S = len(src_patches) # number of sources
#  cdef int dim0 = image_data.shape[0]
#  cdef int dim1 = image_data.shape[1]
#
#  # cdef double *probs = <double *>malloc((S+1) * sizeof(double))
#  cdef double [:] probs = np.zeros(S+1, dtype=np.double)
#  cdef double num_probs, curr_prob
#  cdef long [:] samples = np.zeros(S+1, dtype=np.int)
#  cdef long curr_sample, unsampled
#
#  cdef int i, j, s
#  cdef long xij
#  cdef NativePatch p
#  for i in range(dim0):
#    for j in range(dim1):
#      sum_probs = probs[S] = eta
#      for s in range(S):
#        # For fast access to arrays of extension types:
#        # https://groups.google.com/d/topic/cython-users/EVkjrW-vwVo
#        # TODO: src_patches[s] calls __Pyx_GetItemInt_List which calls PyList_GET_ITEM,
#        # so this could be faster if we made it a native array of pointers somehow.
#        p = <NativePatch>src_patches[s]
#        if j >= p.x0 and j < p.x1 and i >= p.y0 and i < p.y1:
#          probs[s] = curr_prob = p.data[j - p.x0, i - p.y0]
#          sum_probs += curr_prob
#        else:
#          probs[s] = curr_prob = 0
#
#      # This allocates the image data to each of the samples[:]
#      # using the conditional binomial method used by Numpy and GSL;
#      # see C.S. Davis, The computer generation of multinomial random variates, 1993.
#      unsampled = long(image_data[i, j])
#      for s in range(S+1):
#        curr_prob = probs[s]
#        curr_sample = c_rk_binomial(state, unsampled, curr_prob/sum_probs)
#        samples[s] += curr_sample
#        unsampled -= curr_sample
#        if unsampled <= 0:
#          break
#        sum_probs -= curr_prob
#      if unsampled > 0:
#        samples[S] += curr_sample
#
#  return samples

