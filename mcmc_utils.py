import numpy as np
import re
from tractor import *
from tractor import sdss as tractor_sdss
from astrometry.sdss import * #DR7, band_name, band_index
import time
from scipy.special import gammaln
from scipy.ndimage.interpolation import shift
from celeste import *

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

def getFirstImgWithBand(imgs, bandname):
  for img in imgs:
    if imgs.band == bandname:
      return img
  raise Exception("Tractor must have an image in band %s" % bandname)

def initializeSources(srcs, img, percentile=99):
  data = img.img
  from ndimage_utils import generate_peaks

  for x, y in generate_peaks(data, threshold=np.percentile(data, percentile)):
    pos = img.pixel2equa([x, y])
    kwargs = {}
    for band in SDSS_BANDNAMES:
      kwargs[band] = data[y, x]
    # print kwargs
    # TODO(albertwu): how do we measure temperature?
    srcs.append(PointSrcParams(pos, kwargs, 0))

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

