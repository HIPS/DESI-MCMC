import numpy as np
import matplotlib.pyplot as plt

import pickle
import random

import tractor.sdss as sdss
import astrometry.sdss as asdss
import astrometry.util.fits as aufits

from CelestePy.util.data.get_data import tractor_src_to_celestepy_src
from CelestePy.celeste import FitsImage, celeste_likelihood
from CelestePy.celeste_src import SrcParams

ITERATIONS = 5

STAR_FLUXES_FILE = 'star_fluxes_mog.pkl'
GAL_FLUXES_FILE = 'gal_fluxes_mog.pkl'
GAL_SHAPE_FILE = 'gal_shape_mog.pkl'

BANDS = ['u', 'g', 'r', 'i', 'z']

def make_fits_images(run, camcol, field):
    """gets field files from local cache (or sdss), returns UGRIZ dict of 
    fits images"""
    print """==================================================\n\n
            Grabbing image files from the cache.
            TODO: turn off the tractor printing... """

    imgs = {}
    for band in BANDS:
        print "reading in band %s" % band
        imgs[band] = sdss.get_tractor_image_dr9(run, camcol, field, band)

    fn = asdss.DR9().retrieve('photoField', run, camcol, field)
    F = aufits.fits_table(fn)

    # convert to FitsImage's
    imgfits = {}
    for iband,band in enumerate(BANDS):
        print "converting images %s" % band
        frame   = asdss.DR9().readFrame(run, camcol, field, band)
        calib   = np.median(frame.getCalibVec())
        gain    = F[0].gain[iband]
        darkvar = F[0].dark_variance[iband]
        sky     = np.median(frame.getSky())

        imgfits[band] = FitsImage(iband,
                                  timg=imgs[band],
                                  calib=calib,
                                  gain=gain,
                                  darkvar=darkvar,
                                  sky=sky)
    return imgfits

def propose_from_gal_prior(loc, fluxes_prior, shape_prior):
    re_prior, ab_prior, phi_prior = shape_prior
    fluxes = fluxes_prior.sample()[0]

    # do reverse transformation
    re = re_prior.sample()[0]
    ab = ab_prior.sample()[0]
    phi = phi_prior.sample()[0]

    return SrcParams(loc,
                     a      = 1,
                     v      = loc,
                     theta  = 0.5,
                     phi    = 180 * np.exp(phi) / (1 + np.exp(phi)),
                     sigma  = np.exp(re),
                     rho    = np.exp(ab) / (1 + np.exp(ab)),
                     fluxes = fluxes)

def propose_from_star_prior(loc, prior):
    fluxes = prior.sample()[0]
    return SrcParams(loc,
                     a      = 0,
                     fluxes = fluxes)

def calc_transition_probs(star, galaxy,
                          star_prior, gal_fluxes_prior, gal_shape_prior,
                          images,
                          to_star=True):

    mult = -1
    if to_star:
        mult = 1

    star_ell = np.sum([celeste_likelihood([star], image) for image in images])
    gal_ell  = np.sum([celeste_likelihood([galaxy], image) for image in images])

    prior = star_prior.score([star.fluxes])[0] \
             - gal_fluxes_prior.score([galaxy.fluxes])[0] \
             - gal_shape_prior[0].score([np.log(galaxy.sigma)])[0] \
             - gal_shape_prior[1].score([np.log(galaxy.rho / (1 - galaxy.rho))])[0] \
             - gal_shape_prior[2].score([np.log(galaxy.phi / (180 - galaxy.phi))])[0]

    return np.min(np.exp(mult * (star_ell - gal_ell + prior)), 1)

if __name__=="__main__":
    star_file       = open(STAR_FLUXES_FILE, 'rb')
    gal_fluxes_file = open(GAL_FLUXES_FILE, 'rb')
    gal_shape_file  = open(GAL_SHAPE_FILE, 'rb')

    star_prior       = pickle.load(star_file)
    gal_fluxes_prior = pickle.load(gal_fluxes_file)
    gal_shape_prior  = pickle.load(gal_shape_file)

    star_file.close()
    gal_fluxes_file.close()
    gal_shape_file.close()

    # extract a single source
    run = 125
    camcol = 1
    field = 17
    tsrcs = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgfits = make_fits_images(run, camcol, field)

    # list of images, list of celeste sources
    imgs = [imgfits[b] for b in BANDS]
    srcs = [tractor_src_to_celestepy_src(s) for s in tsrcs]

    src = srcs[1]
    counter = 0
    for i in range(ITERATIONS):
        print "iteration", i
        if src.a == 1:
            star = propose_from_star_prior(src.u, star_prior)
            prob = calc_transition_probs(star, src,
                                         star_prior, gal_fluxes_prior, gal_shape_prior,
                                         imgs)
            print "acceptance prob to star", prob
            if random.random() < prob:
                src = star
                counter += 1
                print "transition to star!"
        elif src.a == 0:
            gal = propose_from_gal_prior(src.u, gal_fluxes_prior, gal_shape_prior)
            prob = calc_transition_probs(src, gal,
                                         star_prior, gal_fluxes_prior, gal_shape_prior,
                                         imgs, to_star=False)
            print "acceptance prob to galaxy", prob
            if random.random() < prob:
                src = gal
                counter += 1
                print "transition to galaxy!"
        else:
            break

    print "number of transitions:", counter

