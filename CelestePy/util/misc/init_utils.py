import os, sys, os.path
import numpy as np
from scipy.ndimage import filters
import CelestePy.planck as planck
from CelestePy import FitsImage, SrcParams
import fitsio
from numpy.lib.stride_tricks import as_strided as ast

def load_imgs_and_catalog(fits_cat_glob):
    """ return a list of images and initialized sources """
    us = []
    teff_catalog = []  # if the image has a temperature - hook it up to one source
    srcs = []
    imgs = []
    for c in fits_cat_glob:

        # figure out RA_DEC for image filename
        ra_decs = os.path.basename(os.path.splitext(c)[0]).split('cat-')[1]
        us.append(ra_decs)

        # load all images
        cat_dir = os.path.dirname(c)
        for band in planck.bands: 
            imgs.append(FitsImage(band, cat_dir + '/stamp-%s-%s.fits'%('%s', ra_decs)))

        # load catalog sources
        # TODO: replace with some other initialization of sources (based on bright spots duh)
        srcs_c = get_sources_from_catalog(cat_dir + '/cat-%s.fits'%ra_decs)
        srcs.extend(srcs_c)

        # figure out which source is at the center
        vs       = np.array([imgs[-1].equa2pixel(srcs_c[s].u) for s in range(len(srcs_c))])
        dists    = np.sum((vs - imgs[-1].rho_n)**2, axis=1)
        temps    = np.zeros(len(vs))
        if 'T_EFF' in imgs[-1].header.keys():
            temps[dists.argmin()] = imgs[-1].header['T_EFF']
        teff_catalog = np.concatenate((teff_catalog, temps))

    return srcs, imgs, teff_catalog, us


def get_sources_from_catalog(cat_file):
    """ Takes a catalog fits file and returns a python list of SrcParam Objects. 
        NOTE: The fits files store these brightness parameters in nanomaggies - they
        need to be adjusted by the _IMAGE SPECIFIC_ calibration parameter when they 
        enter into the likelihood. 
    """
    cat_data = fitsio.read(cat_file)
    cat_header = fitsio.read_header(cat_file)
    keys = ['u', 'g', 'r', 'i', 'z']
    catalog_srcs = []
    for src_info in cat_data: 
        src_info = [s for s in src_info]
        src = SrcParams(u      = np.array(src_info[0:2]),
                        fluxes = dict(zip(keys, src_info[2:])),
                        header = cat_header)
        if np.any(np.array(src.fluxes.values()) < 0):
            continue
        catalog_srcs.append(src)
    return catalog_srcs

def init_sources_from_image_block(img_block): 
    """ Takes in a set of ALIGNED images, and initializes sources on bright
        spots 
    """
    # add all of the pixels together, smoothed
    added_pixels = np.zeros(img_block[0].nelec.shape)
    for img in img_block: 
        added_pixels += filters.gaussian_filter(img.nelec, sigma=2, mode='nearest')

    # compute spread and threshold 
    spread = np.percentile(added_pixels, 70) - np.percentile(added_pixels, 2)
    threshold = np.median(added_pixels) + 3*spread

    # location of peaks in 5x5 windows
    peaks = []
    i = 0
    for h in range(3, added_pixels.shape[0]-3):
        for w in range(3, added_pixels.shape[0]-3):
            local_max = np.max(added_pixels[h-2:h+2, w-2:w+2])
            if added_pixels[h, w] > threshold and \
               added_pixels[h, w] > local_max - .1:
                peaks.append([h+1, w+1])

    # initialize sources, all as stars for now
    srcs = []
    for peak in peaks:
        src = SrcParams(u = img_block[0].pixel2equa(peak),
                        a = 0,
                        b = 1e-9 * np.random.rand(),
                        t = 9000 * np.random.rand() + 1000)
        srcs.append(src)
    return srcs


def init_random_galaxy(u, fluxes=None): 
    if fluxes is None:
        bands = ['u', 'g', 'r', 'i', 'z']
        fluxes = {}
        for b in bands: 
            fluxes[b] = 1e-8 * np.random.rand()
    return SrcParams(u      = u,
                     a      = 1,
                     theta  = np.random.rand(),
                     phi    = np.pi * np.random.rand(),
                     sigma  = 5*np.random.randn()**2,
                     rho    = 5*np.random.randn()**2,
                     fluxes = fluxes)


def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions (pulled from
    John Vinyard's blog: http://www.johnvinyard.com/blog/?p=268 )

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
        a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


