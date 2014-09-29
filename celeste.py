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
from util.gmm_like import gmm_log_like

def gen_model_image(srcs, image):
    """ gen_model_image: computes pixel-wise mean count values from only point sources

        Input:
          - srcs  : python list of PointSrcParam objects
          - image : FitsImage object (currently just a _SINGLE_ band) 

        Output: 
          - model_image : mean count values

        Author: Andy Miller <acm@seas.harvard.edu>
    """
    f_s = np.zeros((len(srcs), image.shape[0], image.shape[1]))
    for s, src in enumerate(srcs):

        # compute pixel space location of source
        # returns the X,Y = Width, Height pixel coordinate corresponding to u
        v_s = image.equa2pixel(src.u) + np.array([0, -2])

        # TODO: turn this into a convolution or something...
        # compute pixel space location, v_{n,s}
        y_grid     = np.arange(image.nelec.shape[0])
        x_grid     = np.arange(image.nelec.shape[1])
        yy, xx     = np.meshgrid(x_grid, y_grid, indexing='xy')
        f_s[s,:,:] = np.exp(gmm_log_like(np.column_stack((xx.ravel(), yy.ravel())) - v_s,
                                         image.weights,
                                         image.means,
                                         image.covars).reshape(xx.shape)).T

        # slow for sanity check
        #for x in range(image.nelec.shape[1]): 
        #    for y in range(image.nelec.shape[0]):
        #        pix_val = np.exp(gmm_log_like(np.array([[x,y]]),
        #                                      image.weights,
        #                                      v_s + image.means,
        #                                      image.covars))
        #        assert np.abs(pix_val - f_s[s,y,x]) < 1e-5

        # compute the band-specific temperature multiplier for this source
        ## I_band_s = 
        ## f_s[s,:,:] *= I_band_s

        # multiply by band-specific brightness.  Brightness values are 
        # stored in nanomaggies, so divide by the image calibration to 
        # convert to counts (acm)
        f_s[s,:,:] *= (src.b[image.band] / image.calib)

    # Plug f_s and I_band into 
    return image.kappa * (image.epsilon + np.sum(f_s, axis=0))


def celeste_likelihood(srcs, image):
    """ Evaluates the likelihood of a set of hypothesis sources given an image """
    lambdas = gen_model_image(srcs, image)
    return np.sum(image.nelec[:] * np.log(lambdas) - lambdas)  #Poisson Likelihood


class PointSrcParams(): 
    """ Point source parameter object. 
        Input: 
          u : 2-d np.array holdin right ascension and declination 
          b : python dictionary such that b['r'] = brightness value for 'r' band
          t : temperature of source
    """
    def __init__(self, u, b, t): 
        self.u = u
        self.b = b
        self.t = t


def get_sources_from_catalog(cat_file):
    """ Takes a catalog fits file and returns a python list of PointSrcParam Objects. 
        NOTE: The fits files store these brightness parameters in nanomaggies - they
        need to be adjusted by the _IMAGE SPECIFIC_ calibration parameter when they 
        enter into the likelihood. 
    """
    cat_data = fitsio.read(cat_file)
    keys = ['u', 'g', 'r', 'i', 'z']
    catalog_srcs = []
    for src_info in cat_data: 
        src_info = [s for s in src_info]
        src = PointSrcParams(u = src_info[0:2], 
                             b = dict(zip(keys, src_info[2:])), 
                             t = None)
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
    def __init__(self, band, fits_file_template="../data/blobs/stamp-%s-130.1765-52.7501.fits"): 
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
        # astrometry wcs object for pixel x,y to equa ra,dec conversion
        self.wcs = WCS(self.band_file) #Tan(self.band_file)

        # set image specific KAPPA and epsilon 
        self.kappa   = header['GAIN']     # TODO is this right??
        self.epsilon = header['SKY']
        self.darkvar = header['DARKVAR']  # also eventually contributes to mean?
        self.calib   = header['CALIB']    # dn = nmaggies / calib, calib is NMGY

        # point spread function
        # TODO: Weird things happening with point spread function
        #   - Are the x/y in the mean0x, var0xx, etc. in the header file 
        #     really x,y?  Or are they reversed? They seem reversed... for 
        #     instance, var0[x,x] and var0[y,y] are definitely reversed with
        #     respect to observed spread
        #   - The covariance between x and y also seem to be negated in order 
        #     to yield patterns similar to the observed behavior.  
        #
        psfvec = [header['PSF_P%d'%i] for i in range(18)]
        self.weights = np.array(psfvec[0:3])
        self.means   = np.array(psfvec[3:9]).reshape(3, 2)  # one comp mean per row
        tmp = self.means[:,0]
        self.means[:,0] = self.means[:,1]
        self.means[:,1] = tmp
        self.means[:,0] *= -1.
        covars       = np.array(psfvec[9:]).reshape(3, 3)   # [var_k(x), var_k(y), cov_k(x,y)] per row
        self.covars  = np.zeros((3, 2, 2))
        for i in range(3):
            self.covars[i,:,:] = np.array([[ covars[i,1], -covars[i,2]],
                                           [-covars[i,2],  covars[i,0]]])

    def equa2pixel(self, s_equa):
        x, y = self.wcs.wcs_world2pix(s_equa[0], s_equa[1], 1)
        return np.array([x, y])

    def pixel2equa(self, s_pixel):
        r, d = self.wcs.wcs_pix2world(s_pixel[0], s_pixel[1], 1)
        return np.array([r, d]) 



