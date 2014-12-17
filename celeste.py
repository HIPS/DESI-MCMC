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
from scipy.misc import logsumexp
from gmm_like import gmm_like
#from gmm_like_fast import gmm_like_2d_covinv_logdet as fast_gmm_like
from planck import photons_expected, photons_expected_brightness
from fits_image import FitsImage

def gen_src_image(src, image, pixel_grid = None):
    """ Generates expected photon image for a single point source.  Multiple
        model images (and the 'sky' term) will add together to form an
        expected image for a single observed image.
          - src   : single PointSrcParam object
          - image : FitsImage object
    """
    # 0. Compute expected photon count for this image from source
    if src.b is not None and src.t is not None:
        expected_photons = photons_expected_brightness(src.t, src.b, image.band)
    elif src.fluxes is not None:
        expected_photons = image.kappa * src.fluxes[image.band] # / image.calib
    else:
        raise Exception("No way to compute expected photons without at least fluxes or brightness")

    # compute pixel space location of source
    f_s = gen_point_source_psf_image(src.u, image, pixel_grid=pixel_grid)
    return f_s * expected_photons

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

class PointSrcParams():
    """ Point source parameter object
        Input:
          u : 2-d np.array holdin right ascension and declination
          b : Total brightness/flux.  Equal to 

               b = ell / (4 * pi d^2)

               where ell is the luminosity (in Suns) and d is the distance
               to the source (in light years)

          fluxes : python dictionary such that b['r'] = brightness value for 'r'
              band.  Note that this is essentially the expected number
              of photons to enter the lens and be recorded by a given band
              over the length of one exposure (typically 1.25^2 meters^2 size
              lens and 54 second exposure)

              This will be kept in nanomaggies (must be scaled by 
              image calibration). If this is present, it takes priority
              over the combo of the next few parameters

          t : effective temperature of source (in Kelvin)
          ell : luminosity of source (in Suns)
          d : distance to source (in light years)
    """
    def __init__(self, u, fluxes=None, b=None, t=None, ell=None, d=None, header=None):
        self.u      = u
        self.b      = b
        self.fluxes = fluxes
        self.t = t
        self.ell = ell
        self.d = d
        self.header = header

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.u, other.u) and self.b == other.b
        else:
            return False

