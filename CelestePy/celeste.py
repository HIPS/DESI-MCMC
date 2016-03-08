# Author: Andrew Miller <acm@seas.harvard.edu>
#
#  This file includes classes and functions to 
#    1) take raw fits files corresponding to different bands 
#       with particular header (TODO: what kind of header?)
#       and compute number of observed electrons per pixel, a Poisson 
#       distributed quantity
#    2) Compute the likelihood under the Celeste probabilistic model
#       that is a function of star brightness (in each band), location 
#       (right ascension and declination), and temperature
#

import autograd.numpy as np
from autograd import grad
from autograd.scipy.misc import logsumexp
import celeste_galaxy_conditionals as gal_funs
from planck import photons_expected, photons_expected_brightness
from fits_image import FitsImage
import mixture_profiles as mp
from util.like import gmm_like_2d

## photometric bands we can handle
BANDS = np.array(['u', 'g', 'r', 'i', 'z'], dtype=object)

def gen_src_image(src, image, return_patch=True):
    """ Generates expected photon image for a single point source.  Multiple
        model images (and the 'sky' term) will add together to form an
        expected image for a single observed image.
          - src   : single SrcParam object
          - image : FitsImage object
    """

    # 0. switch on source type - it's either a star, galaxy, or quasar 
    if src.a == 0:    # star

        # compute expected photons in this band
        if src.t:
            expected_photons = photons_expected_brightness(src.t, src.b, image.band)
        else:
            expected_photons = image.nmgy2counts(src.fluxes[image.band])

        # generate a point source image, and disperse the photons about the image
        f_s,_,_ = gen_point_source_psf_image(src.u, image, return_patch=return_patch)
        return f_s * expected_photons

    elif src.a == 1:  # galaxy
        # expected number of photons in this band is given by the flux value
        f_s,_,_ = gen_galaxy_psf_image(src, image, return_patch=return_patch)
        return f_s * image.nmgy2counts(src.fluxes[image.band])

    elif src.a is None and src.fluxes is not None:
        #TODO: rid all of this code of Nanomaggy to photon count (kappa) variables - 
        # store all fluxes as photon counts
        expected_photons = image.kappa * src.fluxes[image.band]

    else:
        raise Exception("No way to compute expected photons without at least fluxes or brightness")

    # compute pixel space location of source
    f_s,_,_ = gen_point_source_psf_image(src.u, image, return_patch=return_patch)
    return f_s * expected_photons

def gen_src_psf_image(src, image):
    if src.a == 0:
        return gen_point_source_psf_image(src, image)
    elif src.a == 1:
        return gen_galaxy_psf_image(src, image)
    else:
        raise "not implemented!"

def gen_point_source_psf_image_with_fluxes(src_params, fits_image, return_patch=True, psf_grid=None):
    """create point source psf image, using flux values instead of a star model """
    #TODO resolve the src_params.fluxes order vs dict!
    # Re-write these image functions with more precise parameterization... 
    # let the Source/Field/Celeste model classes handle the bookkeeping
    src_img, ylim, xlim  = \
        gen_point_source_psf_image(src_params.u, fits_image,
                                   return_patch=True, psf_grid=psf_grid)
    flux     = src_params.flux_dict[fits_image.band]
    src_img *= (flux / fits_image.calib) * fits_image.kappa
    return src_img, ylim, xlim

def gen_src_image_with_fluxes(src, img):
    """ generic src image with fluxes (as opposed to other appearance
    models e.g. temperature/lum, embedding, etc) """
    if src.a == 0:
        f_s, ylim, xlim = gen_point_source_psf_image_with_fluxes(src, img)
    elif src.a == 1:
        psf_img, ylim, xlim = gal_funs.gen_galaxy_psf_image(
                th  = [src.theta, src.sigma, src.phi, src.rho],
                u_s = src.u,
                img = img)
        gal_flux = (src.flux_dict[img.band] / img.calib ) * img.kappa
        f_s      = gal_flux * psf_img
    return f_s, ylim, xlim


def gen_galaxy_psf_image(src, image, return_patch=True, check_overlap=True):
    """
    TODO: incorporate proper flux scaling
    generates a PSF Image (assigns density values to pixels) for 
    a galaxy source (computes MoG resulting from convolving an MoG with 
    another MoG)
    """
    assert src.a == 1, "generating glaxay psf image for non galaxy."
    th = np.array([src.theta, src.sigma, src.phi, src.rho])
    return gal_funs.gen_galaxy_psf_image(th, src.u, image,
                                         check_overlap = check_overlap,
                                         unconstrained = False,
                                         return_patch = return_patch)


def gen_point_source_psf_image(
        u,                         # source location in equatorial coordinates
        image,                     # FitsImage object
        xlim          = None,      # compute model image only on patch defined
        ylim          = None,      #   by xlim ylimcompute only for this patch
        check_overlap = True,      # speedup to check overlap before computing
        return_patch  = True,      # return the small patch as opposed to large patch (memory/speed purposes)
        psf_grid      = None,      # cached PSF grid to be filled out
        pixel_grid    = None       # Nx2 matrix of discrete pixel values to evaluate mog at
        ):
    """ generates a PSF image (assigns density values to pixels) """
    # compute pixel space location of source
    # returns the X,Y = Width, Height pixel coordinate corresponding to u

    # compute pixel space location, v_{n,s}
    v_s = image.equa2pixel(u)
    does_not_overlap = check_overlap and \
                       (v_s[0] < -50 or v_s[0] > 2*image.nelec.shape[0] or
                       v_s[1] < -50 or v_s[0] > 2*image.nelec.shape[1])
    if does_not_overlap:
        return None, None, None

    # create sub-image - make sure it doesn't go outside of field pixels
    if xlim is None and ylim is None:
        bound = image.R
        minx_b, maxx_b = max(0, int(v_s[0] - bound)), min(int(v_s[0] + bound + 1), image.nelec.shape[1])
        miny_b, maxy_b = max(0, int(v_s[1] - bound)), min(int(v_s[1] + bound + 1), image.nelec.shape[0])
        y_grid = np.arange(miny_b, maxy_b, dtype=np.float)
        x_grid = np.arange(minx_b, maxx_b, dtype=np.float)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
        pixel_grid = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))
    else:
        miny_b, maxy_b = ylim
        minx_b, maxx_b = xlim
        if pixel_grid is None:
            y_grid = np.arange(miny_b, maxy_b, dtype=np.float)
            x_grid = np.arange(minx_b, maxx_b, dtype=np.float)
            xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
            pixel_grid = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))
    grid_shape = (maxy_b-miny_b, maxx_b-minx_b)
    psf_grid_small = gmm_like_2d(x       = pixel_grid,
                                 ws      = image.weights,
                                 mus     = image.means + v_s,
                                 sigs    = image.covars)

    # return the small patch and it's bounding box in the bigger fits_image
    if return_patch:
        return psf_grid_small.reshape(grid_shape, order='C'), \
               (miny_b, maxy_b), (minx_b, maxx_b)

    # instantiate a PSF grid 
    if psf_grid is None:
        psf_grid = np.zeros(image.nelec.shape, dtype=np.float)

    # create full field grid
    psf_grid[miny_b:maxy_b, minx_b:maxx_b] = \
        psf_grid_small.reshape(xx.shape, order='C')
    return psf_grid, (0, psf_grid.shape[0]), (0, psf_grid.shape[1])

    # slow for sanity check
    #for x in range(image.nelec.shape[1]): 
    #    for y in range(image.nelec.shape[0]):
    #        pix_val = np.exp(gmm_log_like(np.array([[x,y]]),
    #                                      image.weights,
    #                                      v_s + image.means,
    #                                      image.covars))
    #        assert np.abs(pix_val - f_s[s,y,x]) < 1e-5
    # Might want to insert a check so that f_s.sum() is close to one (or 
    # f_s.sum() * dA is close to one - otherwise some of your point spread
    # mass will fall off the image, and you're neighboring image will probably 
    # have some unexplained boosting of their counts
    #return f_s


def gen_psf_src_image_bound(src, img):
    """ get radius for this source/image pair that encompasses 1-epsilon 
    of photons generated"""
    if src.a == 0:
        return img.R
    else:
        return gal_funs.gen_galaxy_psf_image_bound(src, img)



def gen_model_image(srcs, image):
    """ gen_model_image: computes pixel-wise mean count values from only point sources

        Input:
          - srcs  : python list of PointSrcParam objects
          - image : FitsImage object (currently just a _SINGLE_ band) 

        Output: 
          - model_image : mean count values

        Author: Andy Miller <acm@seas.harvard.edu>
    """
    f_s = np.zeros((image.nelec.shape))
    for s, src in enumerate(srcs):
        f_s += gen_src_image(src, image, return_patch=False)
    # offset image by epsilon
    return image.epsilon + f_s


def gen_src_prob_layers(srcs, img):
    """ for a particular image, generate the probability that each source
    produced the count in each pixel
    """
    src_patches = np.array([gen_src_image(src, img, False) for src in srcs])
    src_sum     = src_patches.sum(axis=0) + img.epsilon # * img.kappa
    src_probs   = np.zeros((src_patches.shape[0]+1, 
                            src_patches.shape[1], 
                            src_patches.shape[2]))
    src_probs[0,:,:] = img.epsilon / src_sum
    for s in range(1, src_patches.shape[0]+1):
        src_probs[s,:,:] = src_patches[s-1,:,:] / src_sum
    return src_probs


def celeste_likelihood(srcs, image):
    """ Evaluates the likelihood of a set of hypothesis sources given an image """
    lambdas = gen_model_image(srcs, image)
    return np.sum(image.nelec * np.log(lambdas) - lambdas)  #Poisson Likelihood


def celeste_likelihood_multi_image(srcs, images):
    """ Full marginal log likelihood - should never decrease 
        Input: 
            srcs: python list of PointSrcParams objects
            imgs: python list of FitsImage objects
    """
    ll = 0
    for img in images: 
        ll += celeste_likelihood(srcs, img)
    return ll

