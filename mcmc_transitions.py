from mcmc_utils import *
from mcmc_utils import SDSS_BANDNAMES
from mcmc_native_utils import *
import numpy as np
import re
from tractor import *
from tractor import sdss as st
from scipy.stats import norm as normal

def sliceSampleSourceSingleAxis(srcs, im, index, axis=0, w=1e-6, m=100, rand=None):
  """
  Performs "stepping out" slice sampling (Neal 2003) on one axis of a source's position.
  axis is in {0,1}. w is the estimate of the typical size of a slice (possible movement).
  m is an integer limiting the size of a slice to $mw$.
  """
  # print "Slice sampling source %s along axis %s" % (index, axis)
  rand = rand or np.random.RandomState()
  src = srcs[index]
  orig_pos_obj = src.u.copy()
  orig_pos = orig_pos_obj.getParams()
  pos_obj = src.u # in-place

  # From this point, we use the notation in Neal (2003).
  # This is the "stepping out" algorithm.
  x0 = orig_pos[axis]
  log_f0 = celeste_likelihood(srcs, im)
  log_y = log_f0 + np.log(rand.rand())

  U = rand.rand()
  L = x0 - w * U
  R = L + w
  V = rand.rand()
  J = np.floor(m*V)
  K = (m-1) - J

  def log_f(x):
    pos_obj[axis] = x
    src.u = pos_obj
    return celeste_likelihood(srcs, im)

  L_iters = R_iters = 0
  while J > 0 and log_y < log_f(L):
    L = L - w
    J = J - 1
    L_iters += 1
    #print "Lowered L to", L
  while K > 0 and log_y < log_f(R):
    R = R + w
    K = K - 1
    R_iters += 1
    #print "Raised R to", R

  # "Shrinkage" procedure
  shrink_its = 100
  xret = x0
  for shrink_it in range(shrink_its):
    U = rand.rand()
    x1 = L + U * (R - L)
    log_f_x1 = log_f(x1)
    if log_y < log_f_x1:
      # New value was set in tractor already in log_f(x1), we're done
      print "Slice on axis %d took %d left, %d right, %d shrinkage iters; gave ll %e" % \
        (axis, L_iters, R_iters, shrink_it, log_f_x1)
      return srcs
    elif x1 < x0:
      L = x1
    else:
      R = x1

  print "Slice sampling failed after %s shrinkages, resetting" % shrink_its
  src.u = orig_pos_obj
  return srcs

def sampleAuxSourceCounts(srcs, img, eta, rand=None):
  """Returns an array of size S+1
  with u summed over pixels for this image"""
  rand = rand or np.random.RandomState()
  src_patches = [NativePatch(gen_model_image([src], img)) for src in srcs]
  image_data = img.nelec
  return nativeSampleAuxSourceCounts(image_data, src_patches, eta, rand)

  # DEAD CODE - DO NOT EAT

  src_extent_data = [(p.getExtent(), p.getImage()) for p in src_patches]
  S = len(srcs)

  probs = np.zeros(S+1) # scratch matrix
  samples = np.zeros(S+1, dtype=np.int)

  for (i,j), xij in np.ndenumerate(image_data):
    xij = int(xij) # rounding takes too long?
    probs[:] = 0
    patch_intersected_img = False
    for s, ((x0,x1,y0,y1), patch_data) in enumerate(src_extent_data):
      # x0,x1,y0,y1 = patch.getExtent() # doing these in the loop is long
      # patch_data = patch.getImage()
      if j >= x0 and j < x1 and i >= y0 and i < y1:
        patch_intersected_img = True
        probs[s] = patch_data[j-x0, i-y0]

    if patch_intersected_img:
      probs[S] = eta
      probs /= np.sum(probs)
      samples += rand.multinomial(xij, probs)
    else:
      # No sources overlapped this patch,
      # so all photons must be attributed to noise.
      samples[S] += xij
  return samples

def gibbsSampleBrightnesses(srcs, img, aPrior, bPrior, eta, rand=None):
  "Sample brightnesses for all sources"
  rand = rand or np.random.RandomState()

  if len(srcs) == 0:
    return srcs

  # Create a multimap of bands to
  # tuples of (img,summed_aux_var_vec) in those bands
  band_imgs_uvecs = dict([(n,[]) for n in SDSS_BANDNAMES])

  print "Sampling auxiliary source counts for image"
  bandname = getBandNameForImage(img)
  uvec = sampleAuxSourceCounts(srcs, img, eta=eta, rand=rand)
  band_imgs_uvecs[bandname].append((img, uvec))

  # print band_imgs_uvecs

  print "Sampling brightness vectors..."

  for s, src in enumerate(srcs):
    # print "src",s
    bright = src.b
    # BrightnessClass = bright.__class__ # probably tractor.NanoMaggies
    new_bright_dict = dict()
    for bandname in SDSS_BANDNAMES:
      # print "bandname",bandname
      a = aPrior
      b = bPrior
      for img, uvec in band_imgs_uvecs[bandname]:
        # print "img", img.name
        lam = gen_model_image([src], img)
        sumlam = np.sum(lam)

        # If we scale each lamba *up* by the photon scaling factor,
        # the mean of the gamma distribution is effectively divided by the
        # (weighted) mean of the photon scaling factors,
        # so that the brightness vector is sampled in units of nanomaggies!

        # TODO: this feature detection isn't working, so just assume we're working in photons
        # if hasattr(img.getPhotoCal(), "getPhotonScalingFactor"):
        # sumlam = sumlam * img.getPhotoCal().getPhotonScalingFactor()

        # print "sumlam %f" % sumlam
        a += uvec[s]
        b += sumlam

      # print "a,b",a,b
      sampled = rand.gamma(a, 1/b)

      new_bright_dict[bandname] = sampled
    # print new_bright_dict

    # We need to explicitly save brightnesses as NanoMaggies,
    # even if the source brightness was of a different type.
    # TODO: Make sure this is setting them as nanomaggies, not as log-scale mags!
    # Possibly use NanoMaggies.fromMags(Mags(**new_bright_dict))?
    src.setBrightness(NanoMaggies(**new_bright_dict))
  return srcs

# Some utility functions for loops

def doMCMC(srcs, im, allowMergeSplit=False, allowBirthDeath=False, iters=1,
           aPrior=1./3, bPrior=1e-4, eta=1e-1,
           sliceW=3e-5, sliceM=20,
           rand=None, cb=None, cb_memo=None):
  rand = rand or np.random.RandomState()
  cb = cb or (lambda srcs, it, logprob, memo: None)

  logprob = celeste_likelihood(srcs, im)

  cb(srcs, 0, logprob, cb_memo)

  for it in xrange(iters):
    with Timing("gibbs") as t:
      gibbsSampleBrightnesses(srcs, im, aPrior=aPrior, bPrior=bPrior, eta=eta, rand=rand)

    with Timing("slice") as t:
      for i in xrange(len(srcs)):
        sliceSampleSourceSingleAxis(srcs, im, i, 0, m=sliceM, w=sliceW, rand=rand)
        sliceSampleSourceSingleAxis(srcs, im, i, 1, m=sliceM, w=sliceW, rand=rand)

    #if allowMergeSplit:
    #  if rand.rand() > 0.5:
    #    splitStar(tractor, rand=rand)
    #  else:
    #    mergeStar(tractor, rand=rand)

    #if allowBirthDeath:
    #  if rand.rand() > 0.5:
    #    birthStar(tractor, rand=rand)
    #  else:
    #    deathStar(tractor, rand=rand)

    logprob = celeste_likelihood(srcs, im)
    cb(srcs, it+1, logprob, cb_memo)
