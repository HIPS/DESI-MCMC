from mcmc_utils import *
from mcmc_utils import SDSS_BANDNAMES
from mcmc_native_utils import *
import numpy as np
import re
from tractor import *
from tractor import sdss as st
from scipy.stats import norm as normal

def sliceSampleSourceSingleAxis(tractor, index, axis=0, w=1e-6, m=100, rand=None):
  """
  Performs "stepping out" slice sampling (Neal 2003) on one axis of a source's position.
  axis is in {0,1}. w is the estimate of the typical size of a slice (possible movement).
  m is an integer limiting the size of a slice to $mw$.
  """
  # print "Slice sampling source %s along axis %s" % (index, axis)
  rand = rand or np.random.RandomState()
  cat = tractor.getCatalog()
  src = cat[index]
  orig_pos_obj = src.getPosition().copy()
  orig_pos = orig_pos_obj.getParams()
  pos_obj = src.getPosition() # in-place

  # From this point, we use the notation in Neal (2003).
  # This is the "stepping out" algorithm.
  x0 = orig_pos[axis]
  log_f0 = getNewLogProb(tractor)
  log_y = log_f0 + np.log(rand.rand())

  U = rand.rand()
  L = x0 - w * U
  R = L + w
  V = rand.rand()
  J = np.floor(m*V)
  K = (m-1) - J

  def log_f(x):
    pos_obj.setParam(axis, x)
    src.setPosition(pos_obj)
    return getNewLogProb(tractor)

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
      return tractor
    elif x1 < x0:
      L = x1
    else:
      R = x1

  print "Slice sampling failed after %s shrinkages, resetting" % shrink_its
  src.setPosition(orig_pos_obj)

  return tractor

def sampleAuxSourceCounts(tractor, img, eta, rand=None):
  """Returns an array of size S+1
  with u summed over pixels for this image"""
  rand = rand or np.random.RandomState()
  srcs = tractor.getCatalog()
  src_patches = [NativePatch(src.getModelPatch(img)) for src in srcs]
  image_data = img.getImage()
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

def gibbsSampleBrightnesses(tractor, aPrior, bPrior, eta, rand=None):
  "Sample brightnesses for all sources"
  rand = rand or np.random.RandomState()

  if len(tractor.getCatalog()) == 0:
    return tractor

  # Create a multimap of bands to
  # tuples of (img,summed_aux_var_vec) in those bands
  band_imgs_uvecs = dict([(n,[]) for n in SDSS_BANDNAMES])
  for img_index, img in enumerate(tractor.getImages()):
    print "Sampling auxiliary source counts for image %d..." % img_index
    bandname = getBandNameForImage(img)
    uvec = sampleAuxSourceCounts(tractor, img, eta=eta, rand=rand)
    band_imgs_uvecs[bandname].append((img, uvec))

  # print band_imgs_uvecs

  print "Sampling brightness vectors..."

  srcs = tractor.getCatalog()
  for s, src in enumerate(srcs):
    # print "src",s
    bright = src.getBrightness()
    # BrightnessClass = bright.__class__ # probably tractor.NanoMaggies
    new_bright_dict = dict()
    for bandname in SDSS_BANDNAMES:
      # print "bandname",bandname
      a = aPrior
      b = bPrior
      for img, uvec in band_imgs_uvecs[bandname]:
        # print "img", img.name
        lam = src.getUnitFluxModelPatch(img, minval=0.).getPatch()
        sumlam = np.sum(lam)

        # If we scale each lamba *up* by the photon scaling factor,
        # the mean of the gamma distribution is effectively divided by the
        # (weighted) mean of the photon scaling factors,
        # so that the brightness vector is sampled in units of nanomaggies!

        # TODO: this feature detection isn't working, so just assume we're working in photons
        # if hasattr(img.getPhotoCal(), "getPhotonScalingFactor"):
        sumlam = sumlam * img.getPhotoCal().getPhotonScalingFactor()

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
  return tractor

def splitChoiceProb(tractor, src_to_split=None, rand=None):
  rand = rand or np.random.RandomState()
  catalog = tractor.getCatalog()
  num_srcs = len(catalog)

  if not src_to_split:
    idx = rand.choice(num_srcs)
    src_to_split = catalog[idx]

  choice_prob = 1./num_srcs
  return src_to_split, choice_prob

def mergeChoiceProb(tractor, srcs_to_merge=None, rand=None):
  rand = rand or np.random.RandomState()
  catalog = tractor.getCatalog()

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
  for src1, src2 in enumerate_pairs(catalog):
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
  bright = np.array(src.getBrightness().getParams())
  return np.prod(bright) * np.sum(bright)

SPLIT_THROW_SD = 1e-7

def splitRandomParamsLogProb(m, g=None):
  return normal.logpdf(m[0] / SPLIT_THROW_SD) + normal.logpdf(m[1] / SPLIT_THROW_SD)

def splitStar(tractor, rand=None):
  rand = rand or np.random.RandomState()

  print "Proposing star split"

  if len(tractor.getCatalog()) == 0:
    print "Catalog empty!"
    return

  orig_prob = getNewLogProb(tractor)
  src, split_choice_prob = splitChoiceProb(tractor)

  print "split_choice_prob %s" % split_choice_prob

  bright = np.array(src.getBrightness().getParams())
  pos = np.array(src.getPosition().getParams())
  B = len(bright) # number of bands

  # Create random parameters (u in Green 2009's notation) controlling the split.
  m = rand.normal(0, SPLIT_THROW_SD, size=2) # should be a 1D vector
  g = rand.rand(B)

  bright1 = bright * g
  bright2 = bright * (1-g)
  pos1 = pos + np.sum(bright2) * m
  pos2 = pos + np.sum(bright1) * m

  print "New positions: %s %s" % (pos1, pos2)

  p1 = src.getPosition().copy()
  p1.setParams(pos1)
  p2 = src.getPosition().copy()
  p2.setParams(pos2)

  b1 = src.getBrightness().copy()
  b1.setParams(bright1)
  b2 = src.getBrightness().copy()
  b2.setParams(bright2)

  src1 = PointSource(p1, b1)
  src2 = PointSource(p2, b2)

  # consider transition
  tractor.removeSource(src)
  tractor.addSource(src1)
  tractor.addSource(src2)

  new_prob = getNewLogProb(tractor)
  src1, src2, merge_choice_prob = mergeChoiceProb(tractor, srcs_to_merge=(src1, src2), rand=rand)

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
    tractor.removeSource(src1)
    tractor.removeSource(src2)
    tractor.addSource(src)
  return tractor

def mergeStar(tractor, rand=None):
  rand = rand or np.random.RandomState()

  print "Proposing star merge"

  if len(tractor.getCatalog()) < 2:
    print "Catalog has fewer than 2 stars!"
    return

  orig_prob = getNewLogProb(tractor)
  src1, src2, merge_choice_prob = mergeChoiceProb(tractor, rand=rand)

  # Create src, which is the source formed by deterministically merging src1 and src2
  bright1 = np.array(src1.getBrightness().getParams())
  bright2 = np.array(src2.getBrightness().getParams())
  pos1 = np.array(src1.getPosition().getParams())
  pos2 = np.array(src2.getPosition().getParams())

  # Form updated brightnesses and positions for the merger by copying prototype from src1
  b = src1.getBrightness().copy()
  bright = bright1 + bright2
  b.setParams(bright1 + bright2)

  p = src1.getPosition().copy()
  pos = 1./np.sum(bright) * (np.sum(bright1) * pos1 + np.sum(bright2) * pos2)
  p.setParams(pos)

  src = PointSource(p, b)

  tractor.removeSource(src1)
  tractor.removeSource(src2)
  tractor.addSource(src)

  new_prob = getNewLogProb(tractor)
  src, split_choice_prob = splitChoiceProb(tractor, src_to_split=src, rand=rand)

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
    tractor.removeSource(src)
    tractor.addSource(src1)
    tractor.addSource(src2)
  return tractor

def birthChoiceProb(tractor, new_src=None, rand=None):
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
  im = tractor.getImages()[0] # we'll use the first image's intensities
  im_data = im.getImage()
  im_height, im_width = im_data.shape
  wcs = im.getWcs()

  had_been_contained = False
  if new_src:
    had_been_contained = tractor.getCatalog().subs.count(new_src)
    if had_been_contained:
      tractor.removeSource(new_src) # will raise error if not had_been_contained

  data_diff = diffData(tractor, im)

  data_diff_sum = np.sum(data_diff)
  data_diff_size = data_diff.size

  if new_src:
    if had_been_contained:
      tractor.addSource(new_src)
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
    pos = wcs.pixelToPosition(j, i)

    # TODO: If we want to randomly choose brightnesses, include that probability
    kwargs = {}
    for band in SDSS_BANDNAMES:
      kwargs[band] = sourceBrightnessEstimate(im, im_data, i, j)
    br = NanoMaggies(**kwargs)
    new_src = PointSource(pos, br)

  j, i = wcs.positionToPixel(new_src.getPosition(), new_src)
  new_src_prob = data_diff[i,j] / data_diff_sum

  return new_src, new_src_prob

def deathChoiceProb(tractor, src_to_kill=None, rand=None):
  rand = rand or np.random.RandomState()
  catalog = tractor.getCatalog()
  num_srcs = len(catalog)

  if not src_to_kill:
    idx = rand.choice(num_srcs)
    src_to_kill = catalog[idx]

  choice_prob = 1./num_srcs
  return src_to_kill, choice_prob

def birthStar(tractor, rand=None):
  rand = rand or np.random.RandomState()
  
  print "Proposing birth starting with %d sources" % len(tractor.getCatalog())
  orig_prob = getNewLogProb(tractor)

  new_src, birth_choice_prob = birthChoiceProb(tractor, rand=rand)

  tractor.addSource(new_src)
  new_prob = getNewLogProb(tractor)

  new_src, death_choice_prob = deathChoiceProb(tractor, src_to_kill=new_src, rand=rand)

  log_alpha = min(0, new_prob - orig_prob + np.log(death_choice_prob) - np.log(birth_choice_prob))
  print "acceptance level for birth", log_alpha
  if np.log(rand.rand()) >= log_alpha:
    print "reverting birth"
    tractor.removeSource(new_src)
  else:
    print "accepted birth"
  print "length of tractor at end of birth:", len(tractor.getCatalog()) 
  return tractor

def deathStar(tractor, rand=None):
  rand = rand or np.random.RandomState()
  print "Proposing death of star"

  if len(tractor.getCatalog()) == 0:
    print "Catalog empty!"
    return

  orig_prob = getNewLogProb(tractor)
  src_to_kill, death_choice_prob = deathChoiceProb(tractor, rand=rand)

  tractor.removeSource(src_to_kill)
  new_prob = getNewLogProb(tractor)

  src_to_kill, birth_choice_prob = birthChoiceProb(tractor, new_src=src_to_kill, rand=rand)

  # accept or reject?
  log_alpha = min(0, new_prob - orig_prob + np.log(birth_choice_prob) - np.log(death_choice_prob))
  print "acceptance level for death", log_alpha
  if np.log(rand.rand()) >= log_alpha:
    print "rejected death"
    tractor.addSource(src_to_kill)
  else:
    print "accepted death"
  return tractor

# Some utility functions for loops

def doMCMC(tractor, allowMergeSplit=False, allowBirthDeath=False, iters=1,
           aPrior=1./3, bPrior=1e-4, eta=1e-1,
           sliceW=3e-5, sliceM=20,
           rand=None, cb=None, cb_memo=None):
  rand = rand or np.random.RandomState()
  cb = cb or (lambda tractor, it, logprob, memo: None)

  logprob = getNewLogProb(tractor)

  cb(tractor, 0, logprob, cb_memo)

  for it in xrange(iters):
    with Timing("gibbs") as t:
      gibbsSampleBrightnesses(tractor, aPrior=aPrior, bPrior=bPrior, eta=eta, rand=rand)

    with Timing("slice") as t:
      for i in xrange(len(tractor.getCatalog())):
        sliceSampleSourceSingleAxis(tractor, i, 0, m=sliceM, w=sliceW, rand=rand)
        sliceSampleSourceSingleAxis(tractor, i, 1, m=sliceM, w=sliceW, rand=rand)

    if allowMergeSplit:
      if rand.rand() > 0.5:
        splitStar(tractor, rand=rand)
      else:
        mergeStar(tractor, rand=rand)

    if allowBirthDeath:
      if rand.rand() > 0.5:
        birthStar(tractor, rand=rand)
      else:
        deathStar(tractor, rand=rand)

    logprob = getNewLogProb(tractor)
    cb(tractor, it+1, logprob, cb_memo)
