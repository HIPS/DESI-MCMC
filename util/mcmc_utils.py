import numpy as np
import re
from tractor import *
from tractor import sdss as tractor_sdss
from astrometry.sdss import * #DR7, band_name, band_index
import time
from scipy.special import gammaln
from scipy.ndimage.interpolation import shift

class Timing:
  def __init__(self, name="(unnamed)"):
    self.name = name
  def __enter__(self):
    self.start = time.clock()
    return self
  def __exit__(self, *args):
    print "Block \"%s\" took %.03f sec" % (self.name, time.clock() - self.start)

def enumerate_pairs(L):
  L = list(L)
  while L:
    i = L.pop()
    for j in L:
      yield (i, j)

SDSS_BANDNAMES = ['u','g','r','i','z']

class PhotonScaledPhotoCal(ScaledPhotoCal):
  """
  When this is used to wrap a PhotoCal instance in a Tractor,
  things like `tractor.getModelImage(img)` will be output in
  terms of photon numbers, not nanomaggies as would normally be the case.
  See its usage in `loadTractorSingleImage`.

  Its custom method `getPhotonScalingFactor` can be used to divide
  photon numbers (from inference) into nanomaggies that can be passed
  into constructors for Brightness instances, etc.
  """
  def getPhotonScalingFactor(self):
    "Nanomaggies are multiplied by this to get photon numbers."
    return self.factor

def loadTractorSingleImage(run=1752, camcol=3, field=164, bands=['r'],
                           roi=[100,600,100,600], dr='dr9', data_dir='data',
                           catalog=False, transform_to_photons=True):
  """
  Returns a Tractor instance for a single region of sky,
  as specified by a specific run-camcol-field identifier.
  Can possibly include multiple bands; note that bands 'r',
  'i', and 'g' must be included in order to render a
  false color image for this region of sky.
  All relevant files are loaded from the relative data_dir,
  downloading from the SDSS data repository if they are not found.
  Pass catalog=True to initialize the tractor with the SDSS
  derived catalog from the specified data release.
  If transform_to_photons is true, the tractor's images and
  their accompanying PhotoCal instances will be set up
  so that everything is in terms of integer photons ("data numbers")
  which we view as the Poisson-distributed quantities.
  """

  # When tractor.sdss.*get_tractor_image* is called without an 
  # astrometry.sdss.common.SdssDR object (from which DR8 etc. inherit),
  # it creates one without arguments, so it by default
  # downloads (sdss_obj.retrieve) FITS files from the web 
  # (sdss_obj.daspaths, sdss_obj.dasurl used in sdss_obj.get_url)
  # and stores them in the working directory
  # (sdss_obj.basedir, used in common.SdssDR code).
  # So if we want to use data from a local FS, or download from a
  # physically closer URL, modify those here once sdss_obj is created.

  imkw = {}
  if dr == 'dr9':
    getim = tractor_sdss.get_tractor_image_dr9
    getsrc = tractor_sdss.get_tractor_sources_dr9
    sdss_obj = DR9(curl=False, basedir=data_dir)
    imkw.update(zrange=[-3,100])
    drnum = 9
  elif dr == 'dr8':
    getim = tractor_sdss.get_tractor_image_dr8
    getsrc = tractor_sdss.get_tractor_sources_dr8
    sdss_obj = DR8(curl=False, basedir=data_dir)
    imkw.update(zrange=[-3,100])
    drnum = 8
  else:
    getim = tractor_sdss.get_tractor_image
    getsrc = tractor_sdss.get_tractor_sources
    sdss_obj = DR7(curl=False, basedir=data_dir)
    imkw.update(useMags=True)
    drnum = 7
    
  sdss_obj.basedir = data_dir

  tims = []
  for bandname in bands:
    tim, info = getim(run, camcol, field, bandname, roi=roi, sdss=sdss_obj, **imkw)
    print info

    if transform_to_photons:
      # info['dn'] is calculated as `dn= img/cimg+simg`
      # as described in "Example of use, and calculating errors" at
      # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
      # There seems to be a discrepancy in this documentation,
      # as these "data numbers" are very close to integers,
      # but the documentation says they still need to be multiplied by a gain number
      # to get to the number of photo-electrons. I assume that the photo-electron
      # conversion was already included in the pipeline
      tim.data = np.round(info['dn'])

      # info['nmgy'] is tne HDU0 NMGY header in the frame FITS file,
      # which is (roughly) the mean of the calibration vector in HDU1,
      # and is used as $ dn * nmgy \approx nanomaggies $.
      # Technically, it's $ (dn - sky) * nmgy \approx nanomaggies $
      # Therefore, we need to scale any catalog brightnesses (in nanomaggies)
      # by the inverse of this calibration value so that they render as an
      # expected number of photons emitted by a point source.
      # This functionality is provided by this image's PhotoCal instance,
      # so we wrap it using the Tractor-provided scaling facility for photocals,
      # with a subclass for bookkeeping purposes.
      tim.photocal = PhotonScaledPhotoCal(tim.photocal, 1./info['nmgy'])

    tim.zr = info['zr']
    #tim.counts = np.round(info['dn'])
    tims.append(tim)

  tractor = Tractor(tims)
  tractor.eta = 120.

  if catalog:
    srcs = getsrc(run, camcol, field, roi=roi, sdss=sdss_obj)
    for src in srcs:
      if isinstance(src, PointSource):
        tractor.addSource(src)

  print "Tractor created with %d images and %d sources" % \
    (len(tractor.getImages()), len(tractor.getCatalog()))

  return tractor

# TODO: we might use the neighborhood around the pixel
def sourceBrightnessEstimate(image, image_data, i, j):
  # return image_data[i,j]
  val = np.max(image_data)
  # if isinstance(image.getPhotoCal(), PhotonScaledPhotoCal):
  val = val / image.getPhotoCal().getPhotonScalingFactor()
  return val

def getFirstImgWithBand(tractor, bandname):
  for img in tractor.getImages():
    if getBandNameForImage(img) == bandname:
      return img
  raise Exception("Tractor must have an image in band %s" % bandname)

def initializeTractor(tractor, threshold=10):
  tractor.setCatalog(Catalog())

  timg = getFirstImgWithBand(tractor, 'r')
  data = timg.getImage()

  from ndimage_utils import generate_peaks

  for x, y in generate_peaks(data, threshold=threshold):
    pos = timg.getWcs().pixelToPosition(x, y)
    kwargs = {}
    for band in SDSS_BANDNAMES:
      kwargs[band] = sourceBrightnessEstimate(timg, data, y, x)
    # print kwargs
    bright = NanoMaggies(**kwargs)
    tractor.addSource(PointSource(pos, bright))

  # for i in range(len(data[:,1])):
  #   for j in range(len(data[1,:])):
  #     if data[i][j] >= 50:
  #       # print 'adding source'
  #       pos = timg.getWcs().pixelToPosition(j, i)
  #       kwargs = {}
  #       for band in bands:
  #         kwargs[band] = sourceBrightnessEstimate(timg, data, i, j)
  #       print kwargs
  #       bright = NanoMaggies(**kwargs)
  #       tractor.addSource(PointSource(pos, bright))

# sdss.py:742 and sdss.py:1070 have:
# name=('SDSS (r/c/f/b=%i/%i/%i/%s)' % (run, camcol, field, bandname))
_bandNameRegex = re.compile(r"([ugriz])\s*\)\s*$", flags=re.IGNORECASE)
def getBandNameForImage(img):
  try:
    return img.getPhotoCal().bandname
  except:
    return _bandNameRegex.search(img.name).group(1)

def getNewLogProb(tractor):
  ll = 0
  eta = tractor.eta # in photon numbers
  for im in tractor.getImages():

    if not hasattr(im, 'data_gln1'):
      im.data_gln1 = gammaln(im.data+1)
    
    # The following calls src.getModelPatch,
    # which calls img.getPhotoCal().brightnessToCounts(self.brightness)
    # which should bring the brightnesses of the sources into photon numbers
    # if the PhotoCal is a PhotonScaledPhotoCal.
    # TODO: Make sure this uses internal caching
    model_counts = tractor.getModelImage(im)

    # Calculate the Poisson likelihood; inlined to be optimal
    # ll += np.sum(poisson.logpmf(im_counts, model_counts + eta))
    rate = model_counts + eta
    k = im.data
    pois_logpmf = k*np.log(rate)
    pois_logpmf -= im.data_gln1 # == gammaln(k+1)
    pois_logpmf -= rate

    ll += np.sum(pois_logpmf)

  # TODO: should we include priors?

  return ll

def tractorBrightnesses(tractor):
  "Utility function to return an array of brightnesses in counts"
  bands = list(set([getBandNameForImage(img) for img in tractor.getImages()]))
  sorted_srcs = sorted(tractor.getCatalog(), key=lambda s: s.getPosition().ra + s.getPosition().dec)
  return [dict(zip(bands,
                   [getFirstImgWithBand(tractor, band).getPhotoCal().brightnessToCounts(src.getBrightness()) \
                    for band in bands]))
          for src in sorted_srcs]

def createImageDifference(filename, im1, im2):
  diff = im2 - im1
    
  from scipy.misc import imsave 
  imsave(filename, np.log(diff - np.min(diff) + 0.5))

def shiftBandImageToRef(img, refimg, data=None):
  if data is None:
    data = img.getImage()

  refpixctr = [n/2 for n in refimg.getShape()]
  refpos = refimg.getWcs().pixelToPosition(*refpixctr)

  from scipy.ndimage.interpolation import shift

  assert img.getShape() == refimg.getShape()

  pixdiff = img.getWcs().positionToPixel(refpos)
  pixshift = (refpixctr[1]-pixdiff[1], refpixctr[0]-pixdiff[0])

  shifted = shift(data, pixshift, order=1,
                  mode='constant', cval=0.0, prefilter=False)

  return shifted

def diffData(tractor, im):
  im_data = im.getImage()
  model_data = tractor.getModelImage(im)
  data_diff = np.clip(im_data - model_data, 0, np.inf)

  # Bias towards only the brightest points
  data_diff[data_diff < np.median(data_diff)] = 0

  return data_diff

def falseColorImage(tractor, useModel=False):
  """
  Returns a PIL image which can be saved with rval.save(fname)
  or shown with matplotlib.pyplot.imshow(rval).
  The image corresponds to the actual image data present in the
  tractor instance if useModel==False,
  OR to the rendered image data from the tractor instance's
  current catalog state if useModel==True.

  Mimics the SDSS3 photoop pipeline stage sdss_frame_jpg.pro,
  which in turn calls djs_rgb_make.pro and polywarp_shift.pro.
  
  Outputs a false color image by mapping the i, r, and g bands
  to the RGB color channels, respectively.
  Because the images can have slight offsets due to the camera
  properties, the i and g bands are shifted so that they
  have the same centers as the r band.
  In the original pipeline, RGB channels are saturated at 30.
  (contiguous saturated pixels are replaced with their average 
  color), then scaled by 0.4 * [5., 6.5, 9.], then put through
  a nonlinear function [r,g,b] *= asinh(N*(r+g+b))/(N*(r+g+b))
  where N = 2.25.

  We simplify this by removing the nonlinearity and saturation.

  (Interesting tidbit from code: it seems there are 1361 unique
  rows in each image.)
  
  For now, we assume there are only images for the specified bands.
  """
  refimg   = getFirstImgWithBand(tractor, 'r')
  # sdss_frame_jpg.pro
  # which does a polywarp shift of degree 1 (unless "keyword" set?)
  # filling with constant 0

  def imgToCorrectedData(img):
    if useModel:
      # acm edit: for false color images don't shift (this seems to saturate 
      #           the final output - this is just for human visualization
      #           anyway, right?  
      data = tractor.getModelImage(img) #+ tractor.eta
    else:
      data = img.getImage()

    # acm edit: the function img.getPhotoCal() seems to be trouble in the 
    #           version of tractor I'm running (maybe Brenton hacked this
    #           method in his version).  Ignoring this seems to produce
    #           sane-looking false color images, though ...
    #
    # If the image was in photons not nanomaggies ("counts"),
    # undo that, since the thresholds below assume the data is in counts.
    # if isinstance(img.getPhotoCal(), PhotonScaledPhotoCal):
    #data = data / img.getPhotoCal().getPhotonScalingFactor()
    return shiftBandImageToRef(img, refimg, data=data)

  idata = (0.4 * 5.) * imgToCorrectedData(getFirstImgWithBand(tractor, 'i'))
  rdata = (0.4 * 6.5) * imgToCorrectedData(refimg)
  gdata = (0.4 * 9.) * imgToCorrectedData(getFirstImgWithBand(tractor, 'g'))

  # print np.max(gdata)

  stacked = np.dstack((idata, rdata, gdata))

  stacked = np.flipud(stacked)

  # Perform saturation on individual pixels
  stacked[stacked < 0.] = 0.
  stacked[stacked > 30.] = 30.

  nonlin = 2.25
  sum_stacked = np.sum(stacked, axis=2)
  nonlin_mult = np.arcsinh(nonlin * sum_stacked) / (nonlin * sum_stacked + 1e-7)
  stacked *= nonlin_mult[:,:,np.newaxis]

  # Perform RGB saturation, which should be minimal
  stacked[stacked > 1.] = 1.
  from scipy.misc import toimage
  fci_image = toimage(stacked / np.max(stacked) * 256.0, channel_axis=2)

  # acm edit: return the bounding box for plotting x,y axes
  ref_bbox = tractor.getBbox(refimg, tractor.getCatalog())
  return fci_image, ref_bbox

def createTestTractor(proto_catalog=[], bands=['r','i','g']):
  """
  Downloads a test frame to get its coordinate system, recenters the catalog on it,
  and renders the catalog to replace the pixel data in that test frame.
  Then, returns a Tractor instance that represents this.

  Example usage in an IPython Notebook, giving an image with two almost-overlapping stars;
  note lower parameters for NanoMaggies give brighter sources:
  ```
    test = createTestTractor([PointSource(RaDecPos(0, 0), NanoMaggies(r=18, i=19, g=18)),
                              PointSource(RaDecPos(0, 1e-3), NanoMaggies(r=19, i=19, g=19))])
    fci = falseColorImage(test)
    fci.save("test.png")
    from IPython.core.display import Image as IPImage
    IPImage("test.png")
  ```
  """
  axes = 2

  # First shift the sources
  center_pos_vec = np.zeros(2)
  for src in proto_catalog:
    center_pos_vec += src.getPosition().getParams()
  center_pos_vec /= (len(proto_catalog) or 1)

  t = loadTractorSingleImage(catalog=False, bands=bands)
  t_img = t.getImages()[0]
  t_shape = t_img.getImage().shape
  t_pos_vec = np.array(t_img.getWcs().pixelToPosition(t_shape[0]/2, t_shape[1]/2).getParams())
  # print t_pos_vec, center_pos_vec

  new_srcs = []
  for src in proto_catalog:
    pos_vec = src.getPosition().getParams()
    new_src = src.copy()
    new_pos_vec = t_pos_vec + pos_vec - center_pos_vec
    new_src.getPosition().setParams(t_pos_vec + pos_vec - center_pos_vec)
    new_srcs.append(new_src)
  t.setCatalog(Catalog(*new_srcs))

  # Now render the images
  for img in t.getImages():
    # TODO: should this include the sky? think it's set to ConstantSky 0 in DR8+ anyways.
    model_img_arr = t.getModelImage(img, sky=True)
    np.copyto(img.getImage(), model_img_arr)
    np.copyto(img.counts, model_img_arr)

  return t
