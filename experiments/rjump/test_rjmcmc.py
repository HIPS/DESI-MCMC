import numpy as np
import matplotlib.pyplot as plt

import pickle
import random

import tractor.sdss as sdss
import astrometry.sdss as asdss
import astrometry.util.fits as aufits

from CelestePy.util.data import make_fits_images
from CelestePy.util.data.get_data import photoobj_to_celestepy_src
from CelestePy.celeste import FitsImage, gen_src_image_with_fluxes
from CelestePy.celeste_src import SrcParams

ITERATIONS = 10000

STAR_FLUXES_FILE = 'star_fluxes_mog.pkl'
GAL_FLUXES_FILE = 'gal_fluxes_mog.pkl'
GAL_SHAPE_FILE = 'gal_shape_mog.pkl'

BANDS = ['u', 'g', 'r', 'i', 'z']

STRIPE_82_DATA_DIR = "../../data/stripe_82_dataset/"
import sys, os
sys.path.append(STRIPE_82_DATA_DIR)
from load_stripe82_square import df_from_fits, create_matched_dataset

TEST = True

def propose_from_gal_prior(star, fluxes_prior, shape_prior,
                           test=False):
    re_prior, ab_prior, phi_prior = shape_prior
    fluxes = fluxes_prior.sample()[0]

    # do reverse transformation
    re = re_prior.sample()[0,0]
    ab = ab_prior.sample()[0,0]
    phi = phi_prior.sample()[0,0]

    # propose star-like shape parameters
    if test:
        phi = 0.01
        rho = 0.01
        sigma = 0.01
    else:
        sigma = np.exp(re)
        rho = np.exp(ab) / (1 + np.exp(ab))
        phi = np.pi * np.exp(phi) / (1 + np.exp(phi))


    return SrcParams(star.params.u,
                     a      = 1,
                     v      = star.params.u,
                     theta  = 0.5,
                     phi    = phi,
                     sigma  = sigma,
                     rho    = rho,
                     fluxes = np.exp(fluxes))

def propose_from_star_prior(gal, prior, test=False):
    fluxes = prior.sample()[0]
    return SrcParams(gal.params.u,
                     a      = 0,
                     fluxes = np.exp(fluxes))

def calc_src_likelihood(src, images, xlims=None, ylims=None):
    ll = 0
    xlims_new = []
    ylims_new = []
    for idx,image in enumerate(images):
        if xlims:
            patch, ylim, xlim = src.compute_model_patch(image,
                                                        xlim=xlims[idx],
                                                        ylim=ylims[idx])
        else:
            patch, ylim, xlim = src.compute_model_patch(image)

        xlims_new.append(xlim)
        ylims_new.append(ylim)

        dpatch = image.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        mpatch = patch + image.epsilon
        ll += np.sum(dpatch * np.log(mpatch) - mpatch)

    return ll, ylims_new, xlims_new

def calc_transition_probs(model_star, model_galaxy,
                          star_prior, gal_fluxes_prior, gal_shape_prior,
                          images,
                          to_star=True,
                          test=False):

    mult = -1
    if to_star:
        mult = 1

    # use the same patch limits for likelihoods
    star_ell, ylims, xlims = calc_src_likelihood(model_star, images)
    gal_ell,      _,     _ = calc_src_likelihood(model_galaxy, images, xlims, ylims)

    star   = model_star.params
    galaxy = model_galaxy.params
    prior  = star_prior.score([star.fluxes])[0] - gal_fluxes_prior.score([galaxy.fluxes])[0]

    sigma_prior = gal_shape_prior[0].score([np.log(galaxy.sigma)])[0]
    rho_prior = gal_shape_prior[1].score([np.log(galaxy.rho / (1 - galaxy.rho))])[0]
    phi_prior = gal_shape_prior[2].score([np.log(galaxy.phi / (np.pi - galaxy.phi))])[0]
    shape_prior = sigma_prior + rho_prior + phi_prior

    if not test:
        prior -= shape_prior

    if mult * (star_ell - gal_ell + prior) > 0:
        return 1
    else:
        return np.exp(mult * (star_ell - gal_ell + prior))

if __name__=="__main__":
    #star_file       = open(STAR_FLUXES_FILE, 'rb')
    #gal_shape_file  = open(GAL_SHAPE_FILE, 'rb')

    #star_prior       = pickle.load(star_file)
    #gal_shape_prior  = pickle.load(gal_shape_file)
    #if TEST:
    #    gal_fluxes_prior = star_prior
    #else:
    #    gal_fluxes_file = open(GAL_FLUXES_FILE, 'rb')
    #    gal_fluxes_prior = pickle.load(gal_fluxes_file)
    #    gal_fluxes_file.close()

    #star_file.close()
    #gal_shape_file.close()

    # extract a single source
    test_primary_fn = os.path.join(STRIPE_82_DATA_DIR, "square_4263_4.fit")
    test_coadd_fn   = os.path.join(STRIPE_82_DATA_DIR, "square_106_4.fit")
    primary_df      = df_from_fits(test_primary_fn)
    coadd_df        = df_from_fits(test_coadd_fn)

    # create a matched dataset - coadd source (ground truth) to 
    # primary sources (baseline)
    primary_matched, coadd_matched, dists = create_matched_dataset(primary_df, coadd_df)

    #################################################
    # look at the breakdown by field
    #################################################
    print "Stripe 82 dataset statistics:"
    field_info = primary_matched[['run', 'camcol', 'field']].drop_duplicates()
    for field in np.sort(field_info.field):
        primary_field_df = primary_matched[primary_matched.field == field]
        num_stars        = np.sum(primary_field_df.type==6)
        num_gals         = np.sum(primary_field_df.type==3)

    ########################################################
    # subselect stripe field 672 - get existing sources
    ########################################################
    run, camcol, field = 4263, 4, 367
    idx = np.where(primary_matched.field == field)[0]
    primary_field_df = primary_matched.iloc[idx]
    coadd_field_df   = coadd_matched.iloc[idx]
    imgfits = make_fits_images(run, camcol, field)

    # list of images, list of celeste sources
    imgs = [imgfits[b] for b in BANDS]

    import CelestePy.model_sources as models
    reload(models)
    model = models.CelesteGMMPrior()
    model.initialize_sources(photoobj_df = primary_field_df)
    model.add_field(img_dict = imgfits)
    bsrcs, bidx = model.get_brightest(object_type='galaxy', num_srcs = 2, return_idx=True)
    model.srcs = bsrcs

    # create a random galaxy
    src = bsrcs[1]
    #src.params.fluxes = np.exp(gal_fluxes_prior.sample()[0])
    src.params.rho = 0.01
    src.params.phi = 0.01
    to_star_propose = 0
    to_star_trans   = 0
    to_gal_propose  = 0
    to_gal_trans    = 0

    # ....
    src.resample_type()

    assert False

    for i in range(ITERATIONS):
        print "iteration", i

        #fig, axarr = plt.subplots(2, 3)
        #model_src.plot(imgs[2], *axarr[0])
        if src.params.a == 1:
            to_star_propose += 1

            star = propose_from_star_prior(src, star_prior, test=TEST)
            star = models.Source(star, model)
            prob = calc_transition_probs(star, src,
                                         star_prior, gal_fluxes_prior, gal_shape_prior,
                                         imgs, test=TEST)
            print "acceptance prob to star", prob
            #new_model_src.plot(imgs[2], *axarr[1])
            if random.random() < prob:
                src = star
                to_star_trans += 1
                print "transition to star!"

        elif src.params.a == 0:
            to_gal_propose += 1

            gal = propose_from_gal_prior(src, gal_fluxes_prior, gal_shape_prior, test=TEST)
            gal = models.Source(gal, model)
            prob = calc_transition_probs(src, gal,
                                         star_prior, gal_fluxes_prior, gal_shape_prior,
                                         imgs, to_star=False, test=True)
            print "acceptance prob to galaxy", prob
            #new_model_src.plot(imgs[0], *axarr[1])
            if random.random() < prob:
                src = gal
                to_gal_trans += 1
                print "transition to galaxy!"
        else:
            break

    print "to-galaxy proposals: %d, transitions: %d" % (to_gal_propose, to_gal_trans)
    print "to-star proposals: %d, transitions: %d" % (to_star_propose, to_star_trans)
