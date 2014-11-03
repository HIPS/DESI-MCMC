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
#

import fitsio
import numpy as np
from scipy.misc import logsumexp
from astropy.wcs import WCS
from gmm_like import gmm_like
from planck import photons_expected, photons_expected_brightness

def gen_src_image(src, image):
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
        expected_photons = image.kappa * src.fluxes[image.band] / image.calib
    else:
        raise Exception("No way to compute expected photons without at least fluxes or brightness")

    # compute pixel space location of source
    f_s = gen_point_source_psf_image(src.u, image)
    return f_s * expected_photons

def gen_point_source_psf_image(u, image, check_overlap=True): 
    """ generates a PSF image (assigns density values to pixels) """
    # compute pixel space location of source
    # returns the X,Y = Width, Height pixel coordinate corresponding to u
    v_s = image.equa2pixel(u)
    if check_overlap and \
        (v_s[0] < -50 or v_s[0] > 2*image.nelec.shape[0] or v_s[1] < -50 or v_s[0] > 2*image.nelec.shape[1]):
       return np.zeros(image.nelec.shape)

    # TODO: turn this into a convolution/speed it uppp
    # compute pixel space location, v_{n,s}
    y_grid = np.arange(image.nelec.shape[0]) + 1
    x_grid = np.arange(image.nelec.shape[1]) + 1
    yy, xx = np.meshgrid(x_grid, y_grid, indexing='xy')
    f_s    = gmm_like(np.column_stack((xx.ravel(), yy.ravel())),
                      image.weights,
                      image.means + v_s,
                      image.covars,
                      invsigs = image.invcovars,
                      logdets = image.logdets).reshape(xx.shape).T
    #f_s    = np.exp(gmm_log_like(np.column_stack((xx.ravel(), yy.ravel())),
    #                             image.weights,
    #                             image.means + v_s,
    #                             image.covars).reshape(xx.shape)).T
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
    return f_s

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
        f_s += gen_src_image(src, image)
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


class FitsImage():
    """ FitsImage - simple organization of fits file images that 
        Dustin has been passing us.  Each FitsImage maintains it's own 
        header information, most importantly: 

         Camera Orientation Information
          - wcs : astropy.wcs object - used to go between pixel and equatorial 
                  coordinates.  Obviates the following three fields (which 
                  are held onto for testing)
          - rho = (rho_x, rho_y)    : pixel reference point
          - phi = (phi_ra, phi_dec) : reference point in equatorial coord
          - Ups = 2x2 matrix : Projection matrix that takes you from pixel to 
                  equa coordinates

         Camera Point Spread Function (modeled as a mixture of gaussians)
          - weights : MoG weights
          - means   : MoG means (X and/or Y seem to be negated in this model
          - covars  : 2x2 Covariance vectors for PSF MoG

         Calibration Information
          - kappa : gain added after the fact
          - epsilon : expected number of electrons (per pixel) not due to 
                      sources that are modeled
          - darkvar : dark variance (TODO: where does this come into the likelihood?)
          - calib   : calibration value (nanomaggies per count) for the image

         Image Signal Information
          - dn = NxM array    : data number array
          - nelec = NxM array : number of electrons corresponding to each 
                                pixel, indexed nelec[y, x]
    """
    def __init__(self, band, fits_file_template="data/stamps/stamp-%s-130.1765-52.7501.fits"): 
        self.band      = band
        self.band_file = fits_file_template%band
        self.img       = fitsio.read(self.band_file)
        header         = fitsio.read_header(self.band_file)
        self.header    = header

        # Compute the number of electrons, resource: 
        # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
        # (Neither of these look like integers)
        self.dn    = self.img / header["CALIB"] + header["SKY"]
        self.nelec = np.round(self.dn * header["GAIN"])
        self.shape = self.nelec.shape

        # reference points
        # TODO: Does CRPIX1 refer to the first axis of self.img ?? 
        self.rho_n = np.array([header['CRPIX1'], header['CRPIX2']])  # PIXEL REFERENCE POINT
        self.phi_n = np.array([header['CRVAL1'], header['CRVAL2']])  # EQUA REFERENCE POINT
        self.Ups_n = np.array([[header['CD1_1'], header['CD1_2']],   # MATRIX takes you into EQUA TANGENT PLANE
                               [header['CD2_1'], header['CD2_2']]])
        self.Ups_n_inv = np.linalg.inv(self.Ups_n)

        # astrometry wcs object for pixel x,y to equa ra,dec conversion
        self.wcs = WCS(self.band_file) #Tan(self.band_file)

        # set image specific KAPPA and epsilon 
        self.kappa   = header['GAIN']     # TODO is this right??
        self.epsilon = header['SKY'] * self.kappa # background rate
        self.epsilon0 = self.epsilon      # background rate copy (for debuggin)
        self.darkvar = header['DARKVAR']  # also eventually contributes to mean?
        self.calib   = header['CALIB']    # dn = nmaggies / calib, calib is NMGY

        # point spread function
        psfvec       = [header['PSF_P%d'%i] for i in range(18)]
        self.weights = np.array(psfvec[0:3])
        self.means   = np.array(psfvec[3:9]).reshape(3, 2)  # one comp mean per row
        covars       = np.array(psfvec[9:]).reshape(3, 3)   # [var_k(x), var_k(y), cov_k(x,y)] per row
        self.covars  = np.zeros((3, 2, 2))
        self.invcovars = np.zeros((3, 2, 2))
        self.logdets   = np.zeros(3)
        for i in range(3):
            self.covars[i,:,:]    = np.array([[ covars[i,0],  covars[i,2]],
                                              [ covars[i,2],  covars[i,1]]])

            # cache inverse covariance 
            self.invcovars[i,:,:] = np.linalg.inv(self.covars[i,:,:])

            # cache log determinant
            sign, logdet = np.linalg.slogdet(self.covars[i,:,:])
            self.logdets[i] = logdet

    def equa2pixel(self, s_equa):
        #### the WCS operation takes forever for some reason... 
        #x, y = self.wcs.wcs_world2pix(s_equa[0], s_equa[1], 1)
        #return np.array([x, y])

        # faster calculation - directly from fits UPS image
        phi1rad = self.phi_n[1] / 180. * np.pi
        s_iwc = np.array([ (s_equa[0] - self.phi_n[0]) * np.cos(phi1rad),
                           (s_equa[1] - self.phi_n[1]) ])
        s_pix = self.Ups_n_inv.dot(s_iwc) + self.rho_n
        #print x,y, s_pix
        return s_pix

    def pixel2equa(self, s_pixel):
        r, d = self.wcs.wcs_pix2world(s_pixel[0], s_pixel[1], 1)
        return np.array([r, d]) 

