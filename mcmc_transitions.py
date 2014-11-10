from mcmc_utils import *
from mcmc_utils import SDSS_BANDNAMES
import numpy as np
import re
from tractor import *
from tractor import sdss as st
from scipy.stats import norm as normal

def removeObjFromArray(array, obj):
  """
  remove object from numpy array by value
  """
  indices = np.where( array == obj )
  np.delete(array, indices)
  return array

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
  orig_pos = src.u.copy()
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
      #print "Slice on axis %d took %d left, %d right, %d shrinkage iters; gave ll %e" % \
      #  (axis, L_iters, R_iters, shrink_it, log_f_x1)
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
  src_patches = [gen_src_image(src, img) for src in srcs]
  image_data = img.nelec

  #return nativeSampleAuxSourceCounts(image_data, src_patches, eta, rand)
  # DEAD CODE - DO NOT EAT

  S = len(srcs)

  probs = np.zeros(S+1) # scratch matrix
  samples = np.zeros((S+1, image_data.shape[0], image_data.shape[1]), dtype=np.int)

  for (i,j), xij in np.ndenumerate(image_data):
    xij = int(xij) # rounding takes too long?
    probs[:] = 0
    # patch_intersected_img = False
    # for s, patch_data in enumerate(src_extent_data):
      # x0,x1,y0,y1 = patch.getExtent() # doing these in the loop is long
      # patch_data = patch.getImage()
      # if j >= x0 and j < x1 and i >= y0 and i < y1:
      #  patch_intersected_img = True
      #  probs[s] = patch_data[j-x0, i-y0]

    probs[0] = img.epsilon
    for k, s in enumerate(src_patches):
        probs[k + 1] = s[i, j]
    probs /= np.sum(probs)
    samples[:,i,j] = rand.multinomial(xij, probs)
    #else:
      # No sources overlapped this patch,
      # so all photons must be attributed to noise.
    #  samples[S] += xij
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
  bandname = img.band
  uvec = sampleAuxSourceCounts(srcs, img, eta=eta, rand=rand).sum(axis=(1, 2))
  band_imgs_uvecs[bandname].append((img, uvec))

  # print band_imgs_uvecs

  print "Sampling brightness vectors..."

  for s, src in enumerate(srcs):
    # print "src",s
    bright = src.b
    new_bright_dict = dict()

    bandname = img.band
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
    src.b = new_bright_dict
  return srcs

# reversible-jump

def splitChoiceProb(srcs, im, src_to_split=None, rand=None):
  rand = rand or np.random.RandomState()
  num_srcs = len(srcs)

  if not src_to_split:
    idx = rand.choice(num_srcs)
    src_to_split = srcs[idx]

  choice_prob = 1./num_srcs
  return src_to_split, choice_prob

def mergeChoiceProb(srcs, im, srcs_to_merge=None, rand=None):
  rand = rand or np.random.RandomState()

  def dist(src1, src2):
    # TODO: implement this
    return 1e-10

  RADIUS = 1e-6 # TODO: define this

  # TODO: Don't do brute-force search!
  if srcs_to_merge:
    srcs_to_merge = tuple(srcs_to_merge)
    sm1, sm2 = srcs_to_merge
  else:
    sm1, sm2 = None, None

  # print sm1, sm2

  close_pairs = []
  # found = False # TODO: These commented lines assumed tuples ignored ordering!
  for src1, src2 in enumerate_pairs(srcs):
    if dist(src1, src2) <= RADIUS:
      close_pairs.append((src1, src2))
      # if sm1 == src1 and sm2 == src2:
        # found = True

  # if srcs_to_merge and not found:
  #   raise Exception("Passed sources were not close")

  # print close_pairs
    
  num_pairs = len(close_pairs)
  if not srcs_to_merge:
    idx = rand.choice(num_pairs)
    srcs_to_merge = close_pairs[idx]
  sm1, sm2 = srcs_to_merge
  choice_prob = 1./num_pairs

  return sm1, sm2, choice_prob

def jacobianForSplit(src):
  bright = src.b
  return np.prod(bright) * np.sum(bright)

SPLIT_THROW_SD = 1e-7
SPLIT_TEMP_SD = 50

START_TEMP = 5000

def splitRandomParamsLogProb(m, g=None):
  return normal.logpdf(m[0] / SPLIT_THROW_SD) + normal.logpdf(m[1] / SPLIT_THROW_SD)

def splitStar(srcs, im, rand=None):
  rand = rand or np.random.RandomState()

  print "Proposing star split"

  if len(srcs) == 0:
    print "Catalog empty!"
    return

  orig_prob = celeste_likelihood(srcs, im) # log likelihood
  src, split_choice_prob = splitChoiceProb(srcs, im)

  print "split_choice_prob %s" % split_choice_prob

  pos = np.array(src.u)
  ell = src.ell
  t = src.t

  # Create random parameters (u in Green 2009's notation) controlling the split.
  m = rand.normal(0, SPLIT_THROW_SD, size=2) # should be a 1D vector
  g = rand.rand()
  h = rand.normal(0, SPLIT_TEMP_SD);

  ell1 = ell * g
  ell2 = ell * (1-g)
  pos1 = pos + ell2 * m
  pos2 = pos - ell1 * m
  t1 = t + ell2 * h
  t2 = t - ell1 * h

  if t1 < 0 or t2 < 0:
    return srcs

  print "New positions: %s %s" % (pos1, pos2)

  p1 = np.array(pos1)
  p2 = np.array(pos2)

  src1 = PointSrcParams(p1, t=t1, ell=ell1)
  src2 = PointSrcParams(p2, t=t2, ell=ell2)

  # consider transition
  removeObjFromArray(srcs, src)
  np.append(srcs, src1)
  np.append(srcs, src2)

  new_prob = celeste_likelihood(srcs, im)
  src1, src2, merge_choice_prob = mergeChoiceProb(srcs, im, srcs_to_merge=(src1, src2), rand=rand)

  print "merge_choice_prob %s" % merge_choice_prob

  jacob = jacobianForSplit(src)
  params_prob = splitRandomParamsLogProb(m, g)

  print "jacob %4e, params %4e" % (np.log(jacob), params_prob)

  print "Log Prob: %4e -> %4e" % (orig_prob, new_prob)

  # This is alpha(x -> x') so g(u) which is params_prob is on the bottom, 
  # and the jacobian is not inverted.
  # split_choice_prob is pi(x) and merge_choice_prob is pi(x')
  log_alpha = min(0, new_prob - orig_prob - params_prob + np.log(jacob) \
                  + np.log(merge_choice_prob) - np.log(split_choice_prob))
  
  print "Split alpha: %s" % log_alpha

  if np.log(rand.rand()) >= log_alpha:
    print "Reverting split"
    removeObjFromArray(srcs, src1)
    removeObjFromArray(srcs, src2)
    np.append(srcs, src)
  return srcs 

def mergeStar(srcs, im, rand=None):
  rand = rand or np.random.RandomState()

  print "Proposing star merge"

  if len(srcs) < 2:
    print "Catalog has fewer than 2 stars!"
    return

  orig_prob = celeste_likelihood(srcs, im)
  src1, src2, merge_choice_prob = mergeChoiceProb(srcs, im, rand=rand)

  # Create src, which is the source formed by deterministically merging src1 and src2
  ell1 = src1.ell
  ell2 = src2.ell
  pos1 = src1.u
  pos2 = src2.u
  t1 = src1.t
  t2 = src2.t

  # Form updated brightnesses and positions for the merger by copying prototype from src1
  ell = ell1 + ell2
  pos = 1./ell * (ell1 * pos1 + ell2 * pos2)
  t = 1/.ell * (ell1 * t1 + ell2 * t2)

  src = PointSrcParams(u, t=t, ell=ell)

  removeObjFromArray(srcs, src1)
  removeObjFromArray(srcs, src2)
  np.append(srcs, src)

  new_prob = celeste_likelihood(srcs, im)
  src, split_choice_prob = splitChoiceProb(srcs, im, src_to_split=src, rand=rand)

  jacob = jacobianForSplit(src)

  # Bookkeeping: this is the value for m that would be needed to split src into src1 and src2
  # if we were going in the split direction.
  m = (pos1 - pos) / np.sum(bright2)
  params_prob = splitRandomParamsLogProb(m)

  # This is alpha(x' -> x) so g(u) which is params_prob is on the top,
  # and the jacobian is inverted.
  # split_choice_prob is pi(x) and merge_choice_prob is pi(x')
  log_alpha = min(0, new_prob - orig_prob + params_prob - np.log(jacob) \
                  + np.log(split_choice_prob) - np.log(merge_choice_prob))

  print "Merge alpha: %s" % log_alpha

  if np.log(rand.rand()) >= log_alpha:
    print "Reverting merge"
    removeObjFromArray(srcs, src)
    np.append(srcs, src1)
    np.append(srcs, src2)
  return srcs

def birthChoiceProb(srcs, im, new_src=None, rand=None):
  """
  If a source is passed, returns it and its location's
  probability of being chosen for a birth proposal
  given the current catalog, not including the passed source.
  This corresponds to the probability of proposing the reverse
  transition of the death of the passed source.

  If a source is not passed, proposes one and
  returns it and its probability of being chosen for a birth proposal
  given the current catalog.
  """
  rand = rand or np.random.RandomState()
  # im = tractor.getImages()[0] # we'll use the first image's intensities
  im_data = im.nelec
  im_height, im_width = im_data.shape

  had_been_contained = False
  if new_src:
    old_size = srcs.size
    srcs = removeObjFromArray(srcs, new_src)
    if not size(srcs) == old_size:
      has_been_contained = True

  data_diff = diffData(srcs, im)

  data_diff_sum = np.sum(data_diff)
  data_diff_size = data_diff.size

  if new_src:
    if had_been_contained:
      np.append(srcs, new_src)
  else:
    # We're proposing a new birth here
    data_diff_flat = data_diff.flatten()
    data_diff_cumsum = np.cumsum(data_diff_flat)
    idx = np.searchsorted(data_diff_cumsum,
                          rand.rand() * data_diff_sum)

    # data_diff_flat_norm = data_diff_flat / data_diff_sum
    # print np.sum(data_diff_flat_norm)
    # data_len_flat = data_diff_flat.size
    # idx = rand.choice(data_len_flat, p=data_diff_flat_norm)
    i, j = idx / im_width, idx % im_width
    print "Birthing at (i,j) = ", i, j
    pos = im.pixel2equa([j, i])

    # TODO: If we want to randomly choose brightnesses, include that probability
    new_src = PointSrcParams(pos, t=START_TEMP, ell=im_data[j,i])

  j, i = im.equa2pixel(new_src.u)
  new_src_prob = data_diff[i,j] / data_diff_sum

  return new_src, new_src_prob

def deathChoiceProb(srcs, im, src_to_kill=None, rand=None):
  rand = rand or np.random.RandomState()
  # catalog = tractor.getCatalog()
  num_srcs = len(srcs)

  if not src_to_kill:
    idx = rand.choice(num_srcs)
    src_to_kill = srcs[idx]

  choice_prob = 1./num_srcs
  return src_to_kill, choice_prob

def birthStar(srcs, im, rand=None):
  rand = rand or np.random.RandomState()
  
  print "Proposing birth starting with %d sources" % len(srcs)
  orig_prob = celeste_likelihood(srcs, im)

  new_src, birth_choice_prob = birthChoiceProb(srcs, im, rand=rand)

  np.append(srcs, new_src)
  new_prob = celeste_likelihood(srcs, im)

  print len(srcs)
  new_src, death_choice_prob = deathChoiceProb(srcs, im, src_to_kill=new_src, rand=rand)

  log_alpha = min(0, new_prob - orig_prob + np.log(death_choice_prob) - np.log(birth_choice_prob))
  print "acceptance level for birth", log_alpha
  if np.log(rand.rand()) >= log_alpha:
    print "reverting birth"
    removeObjFromArray(srcs, new_src)
  else:
    print "accepted birth"
  print "length of tractor at end of birth:", len(srcs) 
  return srcs

def deathStar(srcs, im, rand=None):
  rand = rand or np.random.RandomState()
  print "Proposing death of star"

  if len(srcs) == 0:
    print "Catalog empty!"
    return

  orig_prob = celeste_likelihood(srcs, im)
  src_to_kill, death_choice_prob = deathChoiceProb(srcs, im, rand=rand)

  removeObjFromArray(srcs, src_to_kill)
  new_prob = celeste_likelihood(srcs, im)

  src_to_kill, birth_choice_prob = birthChoiceProb(srcs, im, new_src=src_to_kill, rand=rand)

  # accept or reject?
  log_alpha = min(0, new_prob - orig_prob + np.log(birth_choice_prob) - np.log(death_choice_prob))
  print "acceptance level for death", log_alpha
  if np.log(rand.rand()) >= log_alpha:
    print "rejected death"
    np.append(srcs, src_to_kill)
  else:
    print "accepted death"
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
    #    splitStar(srcs, im, rand=rand)
    #  else:
    #    mergeStar(srcs, im, rand=rand)

    #if allowBirthDeath:
    #  if rand.rand() > 0.5:
    #    birthStar(srcs, im, rand=rand)
    #  else:
    #    deathStar(srcs, im, rand=rand)

    logprob = celeste_likelihood(srcs, im)
    cb(srcs, it+1, logprob, cb_memo)
