# Class for handling images and parameters given in fits files. Implements
# transformation between 
#
# Author: Andrew Miller <acm@seas.harvard.edu>
import fitsio
import numpy as np

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

        ###### DEAD CODE ######################################################
        #astrometry wcs object for pixel x,y to equa ra,dec conversion
        #self.wcs = WCS(self.band_file) #Tan(self.band_file)
        #######################################################################

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

    def contains(self, s_equa, pad = 50):
        """ can this source be seen by this image? 
          s_equa : equatorial locations of the point in the sky in question
          pad    : how many pixels outside of the image do we include in 'contains'?
        """
        v_s = self.equa2pixel(s_equa)
        return (v_s[0] > -pad) and (v_s[0] < self.nelec.shape[0] + pad) and \
               (v_s[1] > -pad) and (v_s[1] < self.nelec.shape[1] + pad)

    def equa2pixel(self, s_equa):
        phi1rad = self.phi_n[1] / 180. * np.pi
        s_iwc = np.array([ (s_equa[0] - self.phi_n[0]) * np.cos(phi1rad),
                           (s_equa[1] - self.phi_n[1]) ])
        s_pix = self.Ups_n_inv.dot(s_iwc) + self.rho_n
        return s_pix

    def pixel2equa(self, s_pixel):
        phi1rad = self.phi_n[1] / 180. * np.pi
        s_iwc   = self.Ups_n.dot(s_pixel - self.rho_n) 
        s_equa = np.array([ s_iwc[0]/np.cos(phi1rad) + self.phi_n[0], 
                            s_iwc[1] + self.phi_n[1] ])
        return s_equa

