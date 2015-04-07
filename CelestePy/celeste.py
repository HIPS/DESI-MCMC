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

import numpy as np
from autograd import grad
from scipy.misc import logsumexp
from util.like.gmm_like import gmm_like
#from gmm_like_fast import gmm_like_2d_covinv_logdet as fast_gmm_like
import celeste_galaxy_conditionals as gal_funs
from planck import photons_expected, photons_expected_brightness
from fits_image import FitsImage
import mixture_profiles as mp

def gen_src_image(src, image, pixel_grid = None):
    """ Generates expected photon image for a single point source.  Multiple
        model images (and the 'sky' term) will add together to form an
        expected image for a single observed image.
          - src   : single SrcParam object
          - image : FitsImage object
    """

    # 0. switch on source type - it's either a star, galaxy, or quasar 
    if src.a == 0:    # star

        # compute expected photons in this band
        expected_photons = photons_expected_brightness(src.t, src.b, image.band)

        # generate a point source image, and disperse the photons about the image
        f_s = gen_point_source_psf_image(src.u, image, pixel_grid=pixel_grid)
        return f_s * expected_photons

    elif src.a == 1:  # galaxy
        # expected number of photons in this band is given by the flux value
        f_s        = gen_galaxy_psf_image(src, image, pixel_grid = pixel_grid)
        return f_s * image.nmgy2counts(src.fluxes[image.band])

    elif src.a is None and src.fluxes is not None:
        #TODO: rid all of this code of Nanomaggy to photon count (kappa) variables - 
        # store all fluxes as photon counts
        expected_photons = image.kappa * src.fluxes[image.band]

    else:
        raise Exception("No way to compute expected photons without at least fluxes or brightness")

    # compute pixel space location of source
    f_s = gen_point_source_psf_image(src.u, image, pixel_grid=pixel_grid)
    return f_s * expected_photons

## cache galaxy profile mixture components
galaxy_profs = [mp.get_exp_mixture(), mp.get_dev_mixture()]

def gen_galaxy_psf_image(src, image, check_overlap=True, pixel_grid = None):
    """ generates a PSF Image (assigns density values to pixels) for 
    a galaxy source (computes MoG resulting from convolving an MoG with 
    another MoG)
    """
    assert src.a == 1, "generating glaxay psf image for non galaxy."
    th = np.array([src.theta, src.sigma, src.phi, src.rho])
    return gal_funs.gen_galaxy_psf_image(th, src.u, image,
                                         check_overlap = check_overlap,
                                         pixel_grid    = pixel_grid,
                                         unconstrained = False)

    #v_s    = image.equa2pixel(src.u)
    #thetas = [src.theta, 1.-src.theta]

    ## compute spatial covariance matrix
    #R = np.array([[np.cos(src.phi), -np.sin(src.phi)],
    #              [np.sin(src.phi), np.cos(src.phi)]])
    #S = np.diag([src.sigma*src.sigma, src.sigma*src.sigma*src.rho*src.rho])
    #W = R.T.dot(S).dot(R)

    ### mixture of 40ish (yeesh) gaussians - instantiate parameters
    #num_components = len(image.weights) * sum([len(gp.amp) for gp in galaxy_profs])
    #weights = np.zeros(num_components)
    #means   = np.zeros((num_components, 2))
    #covars  = np.zeros((num_components, 2, 2))
    #cnt = 0
    #for k in range(len(image.weights)):                 # num PSF Componenets
    #    for i in range(2):                              # two galaxy types
    #        for j in range(len(galaxy_profs[i].amp)):   # galaxy type components
    #            weights[cnt] = image.weights[k] * thetas[i] * galaxy_profs[i].amp[j]
    #            means[cnt, :] = v_s + image.means[k,:]
    #            covars[cnt, :, :] = image.covars[k,:,:] + \
    #                galaxy_profs[i].var[j,:,:].dot(W)
    #            cnt += 1

    ## instantiate a pixel grid if necessary
    #if pixel_grid is None: 
    #    y_grid = np.arange(image.nelec.shape[0], dtype=np.float) + 1
    #    x_grid = np.arange(image.nelec.shape[1], dtype=np.float) + 1
    #    yy, xx = np.meshgrid(x_grid, y_grid, indexing='xy')
    #    pixel_grid = np.column_stack((xx.ravel(), yy.ravel()))

    ### evaluate equation 11-13 in jeff's november writeup
    #psf_grid = gmm_like(x = pixel_grid, 
    #                    ws = weights,
    #                    mus = means,
    #                    sigs = covars)
    #return psf_grid.reshape(image.nelec.shape).T

def gen_point_source_psf_image(
        u,                         # source location in equatorial coordinates
        image,                     # FitsImage object
        check_overlap = True,      # speedup to check overlap before computing
        pixel_grid    = None,      # cached pixel grid
        psf_grid      = None       # cached PSF grid to be filled out
        ):
    """ generates a PSF image (assigns density values to pixels) """
    # compute pixel space location of source
    # returns the X,Y = Width, Height pixel coordinate corresponding to u

    # compute pixel space location, v_{n,s}
    v_s = image.equa2pixel(u)
    if check_overlap and \
        (v_s[0] < -50 or v_s[0] > 2*image.nelec.shape[0] or v_s[1] < -50 or v_s[0] > 2*image.nelec.shape[1]):
       return np.zeros(image.nelec.shape)

    # instantiate a pixel grid if necessary
    if pixel_grid is None: 
        y_grid = np.arange(image.nelec.shape[0], dtype=np.float) + 1
        x_grid = np.arange(image.nelec.shape[1], dtype=np.float) + 1
        yy, xx = np.meshgrid(x_grid, y_grid, indexing='xy')
        pixel_grid = np.column_stack((xx.ravel(), yy.ravel()))

    # instantiate a PSF grid 
    if psf_grid is None:
        psf_grid = np.zeros(pixel_grid.shape[0], dtype=np.float)

    # compute the PSF (just a mixture call)
    #fast_gmm_like(probs  = psf_grid, 
    #              x      = pixel_grid,
    #              ws     = image.weights,
    #              mus    = image.means + v_s,
    #              invsigs = image.invcovars,
    #              logdets = image.logdets)

    # slow python method
    psf_grid = gmm_like(x = pixel_grid, 
                        ws = image.weights,
                        mus = image.means + v_s,
                        sigs = image.covars,
                        invsigs = image.invcovars,
                        logdets = image.logdets)
    return psf_grid.reshape(image.nelec.shape).T

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

def gen_model_image(srcs, image):
    """ gen_model_image: computes pixel-wise mean count values from only point sources

        Input:
          - srcs  : python list of PointSrcParam objects
          - image : FitsImage object (currently just a _SINGLE_ band) 

        Output: 
          - model_image : mean count values

        Author: Andy Miller <acm@seas.harvard.edu>
    """

    # generate pixel grid to be re-used across sources
    y_grid = np.arange(image.nelec.shape[0], dtype=np.float) + 1
    x_grid = np.arange(image.nelec.shape[1], dtype=np.float) + 1
    yy, xx = np.meshgrid(x_grid, y_grid, indexing='xy')
    pixel_grid = np.column_stack((xx.ravel(), yy.ravel()))

    f_s = np.zeros((image.nelec.shape))
    for s, src in enumerate(srcs):
        f_s += gen_src_image(src, image, pixel_grid)
    # offset image by epsilon
    return image.epsilon + f_s

def gen_src_prob_layers(srcs, img):
    """ for a particular image, generate the probability that each source
    produced the count in each pixel
    """
    src_patches = np.array([gen_src_image(src, img) for src in srcs])
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

