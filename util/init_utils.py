import os, sys, os.path
import numpy as np
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('..'), os.path.pardir)))
import planck
from celeste import FitsImage, PointSrcParams
import fitsio

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
        temps[dists.argmin()] = imgs[-1].header['T_EFF']
        teff_catalog = np.concatenate((teff_catalog, temps))

    return srcs, imgs, teff_catalog, us


def get_sources_from_catalog(cat_file):
    """ Takes a catalog fits file and returns a python list of PointSrcParam Objects. 
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
        src = PointSrcParams(u      = np.array(src_info[0:2]),
                             fluxes = dict(zip(keys, src_info[2:])),
                             header = cat_header)
        if np.any(np.array(src.fluxes.values()) < 0):
            continue
        catalog_srcs.append(src)
    return catalog_srcs



