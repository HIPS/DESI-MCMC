import os
import sys
from math import pi, sqrt, ceil, floor
from datetime import datetime

import pyfits
import pylab as plt
import numpy as np
import pandas as pd

from tractor.engine import *
from tractor.basics import *
from tractor.sdss import *

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import setRadecAxes, redgreen
from astrometry.libkd.spherematch import match_radec

from tractor.basics import PointSource
from tractor.galaxy import ExpGalaxy, DevGalaxy, CompositeGalaxy
import tractor.sdss as sdss
import astrometry.sdss as asdss
import astrometry.util.fits as aufits

from CelestePy.celeste import gen_point_source_psf_image, gen_galaxy_psf_image, FitsImage
from CelestePy.celeste_src import SrcParams

import pickle
import fitsio


# constnats 
BANDS = ['u', 'g', 'r', 'i', 'z']

# simple converter between mags and nanomaggies
def mags2nanomaggies(mags):
    return np.power(10., (mags - 22.5)/-2.5)

def nanomaggies2mags(nanos):
    return (-2.5)*np.log10(nanos) + 22.5

#
# Get a UGRIZ dictionary of fits images (only works for DR8+ fields)
#
def make_fits_images(run, camcol, field):
    """gets field files from local cache (or sdss), returns UGRIZ dict of 
    fits images"""
    print """==================================================\n\n
            Grabbing image files from the cache.
            TODO: turn off the tractor printing... """

    # grab photo field objects - cache in 'sdss_data' if possible
    if not os.path.exists('sdss_data'):
        os.makedirs('sdss_data')
    sdssobj = asdss.DR9(basedir='sdss_data')

    # grab images
    imgs = {}
    for band in BANDS:
        print "reading in band %s" % band
        imgs[band] = sdss.get_tractor_image_dr9(run, camcol, field, band, sdss=sdssobj)

    # grab photo field
    fn = sdssobj.retrieve('photoField', run, camcol, field)
    F  = aufits.fits_table(fn)

    # convert to FitsImage's
    imgfits = {}
    for iband,band in enumerate(BANDS):
        print "converting images %s" % band
        frame   = sdssobj.readFrame(run, camcol, field, band)
        calib   = np.median(frame.getCalibVec())
        gain    = F[0].gain[iband]
        darkvar = F[0].dark_variance[iband]
        sky     = np.median(frame.getSky())
        imgfits[band] = FitsImage(band,
                                  timg=imgs[band],
                                  calib=calib,
                                  gain=gain,
                                  darkvar=darkvar,
                                  sky=sky,
				  frame=frame,
				  fits_table=F)
    return imgfits


def photoobj_to_celestepy_src(photoobj_row):
    """Conversion between tractor source object and our source object...."""
    u = photoobj_row[['ra', 'dec']].values

    # brightnesses are stored in mags (gotta convert to nanomaggies)
    mags   = photoobj_row[['psfMag_%s'%b for b in ['u', 'g', 'r', 'i', 'z']]].values
    fluxes = [mags2nanomaggies(m) for m in mags]

    # photoobj type 3 are gals, type 6 are stars
    if photoobj_row.type == 6:
        return SrcParams(u, a=0, fluxes=fluxes)
    else:

        # compute frac dev/exp 
        prob_dev = np.mean(photoobj_row[['fracDeV_%s'%b for b in BANDS]])

        # galaxy A/B estimate, angle, and radius
        devAB      = np.mean(photoobj_row[['deVAB_%s'%b for b in BANDS]])
        devRad     = np.mean(photoobj_row[['deVRad_%s'%b for b in BANDS]])
        devPhi     = np.mean(photoobj_row[['deVPhi_%s'%b for b in BANDS]])
        expAB      = np.mean(photoobj_row[['expAB_%s'%b for b in BANDS]])
        expRad     = np.mean(photoobj_row[['expRad_%s'%b for b in BANDS]])
        expPhi     = np.mean(photoobj_row[['expPhi_%s'%b for b in BANDS]])

        #estimate - mix over frac dev/exp
        AB  = prob_dev * devAB + (1. - prob_dev) * expAB
        Rad = prob_dev * devRad + (1. - prob_dev) * expRad
        Phi = prob_dev * devPhi + (1. - prob_dev) * expPhi

        ## galaxy flux esimates
        devFlux = np.array([mags2nanomaggies(m) for m in
                                photoobj_row[['deVMag_%s'%b for b in BANDS]]])
        expFlux = np.array([mags2nanomaggies(m) for m in
                                photoobj_row[['expMag_%s'%b for b in BANDS]]])
        fluxes = prob_dev * devFlux + (1. - prob_dev) * expFlux

        #theta : exponential mixture weight. (1 - theta = devac mixture weight)
        #sigma : radius of galaxy object (in arcsc > 0)
        #rho   : axis ratio, dimensionless, in [0,1]
        #phi   : radians, "E of N" 0=direction of increasing Dec, 90=direction of increasting RAab
        return SrcParams(u,
                         a      = 1,
                         v      = u,
                         theta  = 1.0-prob_dev,
                         phi    = -1.*Phi, #(Phi * np.pi / 180.), # + np.pi / 2) % np.pi,
                         sigma  = Rad,
                         rho    = AB,
                         fluxes = fluxes)


def tractor_src_to_celestepy_src(tsrc):
    """Conversion between tractor source object and our source object...."""
    pos = tsrc.getPosition()
    u = np.array([p for p in pos])

    # brightnesses are stored in mags (gotta convert to nanomaggies)
    def mags2nanomaggies(mags):
        return np.power(10., (mags - 22.5)/-2.5)

    fluxes = tsrc.getBrightnesses()[0]
    fluxes = [mags2nanomaggies(flux) for flux in fluxes]

    if type(tsrc) == PointSource:
        return SrcParams(u, a=0, fluxes=fluxes)
    else:
        if type(tsrc) == ExpGalaxy:
            shape = tsrc.getShape()
            theta = 1.
        elif type(tsrc) == DevGalaxy:
            shape = tsrc.getShape()
            theta = 0.
        elif type(tsrc) == CompositeGalaxy:
            shape = tsrc.shapeExp
            theta = 0.5
        else:
            pass

        return SrcParams(u,
                         a=1,
                         v=u,
                         theta = theta,
                         phi   = shape[2], #-1*(shape[2]-90), # * np.pi / 180., #(shape[2] + 180) * np.pi / 180,
                         sigma = shape[0],
                         rho   =shape[1],
                         fluxes=fluxes)


def df_from_fits(filename, i=1):
    """ create a pandas dataframe from a fits file """
    return pd.DataFrame.from_records(fitsio.FITS(filename)[i].read().byteswap().newbyteorder())


def loadTractorSingleImage(run=1752, camcol=3, field=164, bands=['r'],
                           roi=[100,600,100,600], dr='dr9', data_dir='data',
                           catalog=False):
    imkw = {}
    if dr == 'dr9':
        getim = get_tractor_image_dr9
        sdss_obj = DR9(curl=False, basedir=data_dir)
        imkw.update(zrange=[-3,100])
        drnum = 9
        print "drnum 9"
    elif dr == 'dr8':
        getim = get_tractor_image_dr8
        sdss_obj = DR8(curl=False, basedir=data_dir)
        imkw.update(zrange=[-3,100])
        drnum = 8
        print "drnum 8"
    else:
        getim = tractor_sdss.get_tractor_image
        getsrc = tractor_sdss.get_tractor_sources
        sdss_obj = DR7(curl=False, basedir=data_dir)
        imkw.update(useMags=True)
        drnum = 7
        print "drnum 7"

    sdss_obj.basedir = data_dir

    tims = []
    for bandname in bands:
        tim, info = getim(run, camcol, field, bandname, roi=roi, sdss=sdss_obj, **imkw)
        print info

def get_tractor_image_dr9(*args, **kwargs):
    sdss = kwargs.get('sdss', None)
    if sdss is None:
        curl = kwargs.pop('curl', False)
        kwargs['sdss'] = DR9(curl=curl)
    return get_tractor_image_dr8(*args, **kwargs)

def get_tractor_image_dr8(run, camcol, field, bandname, sdss=None,
                          roi=None, psf='kl-gm', roiradecsize=None,
                          roiradecbox=None,
                          savepsfimg=None, curl=False,
                          nanomaggies=False,
                          zrange=[-3,10],
                          invvarIgnoresSourceFlux=False,
                          invvarAtCenter=False,
                          invvarAtCenterImage=False,
                          imargs={}):
    # retry_retrieve=True,
    '''
    Creates a tractor.Image given an SDSS field identifier.

    If not None, roi = (x0, x1, y0, y1) defines a region-of-interest
    in the image, in zero-indexed pixel coordinates.  x1,y1 are
    NON-inclusive; roi=(0,100,0,100) will yield a 100 x 100 image.

    psf can be:
      "dg" for double-Gaussian
      "kl-gm" for SDSS KL-decomposition approximated as a Gaussian mixture

      "bright-*", "*" one of the above PSFs, with special handling at
      the bright end.

    "roiradecsize" = (ra, dec, half-size in pixels) indicates that you
    want to grab a ROI around the given RA,Dec.

    "roiradecbox" = (ra0, ra1, dec0, dec1) indicates that you
    want to grab a ROI containing the given RA,Dec ranges.

    "invvarAtCentr" -- get a scalar constant inverse-variance

    "invvarAtCenterImage" -- get a scalar constant inverse-variance
    but still make an image out of it.

    Returns: (tractor.Image, dict)

    dict contains useful details like:
      'sky'
      'skysig'
    '''

    origpsf = psf
    if psf.startswith('bright-'):
        psf = psf[7:]
        brightpsf = True
        print 'Setting bright PSF handling'
    else:
        brightpsf = False

    valid_psf = ['dg', 'kl-gm', 'kl-pix']
    if psf not in valid_psf:
        raise RuntimeError('PSF must be in ' + str(valid_psf))

    if sdss is None:
        sdss = DR8(curl=curl)

    bandnum = band_index(bandname)

    for ft in ['psField', 'fpM']:
        fn = sdss.retrieve(ft, run, camcol, field, bandname)
    fn = sdss.retrieve('frame', run, camcol, field, bandname)

    # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
    frame = sdss.readFrame(run, camcol, field, bandname, filename=fn)

    #image = frame.getImage().astype(np.float32)
    #(H,W) = image.shape

    H,W = frame.getImageShape()
    
    info = dict()
    hdr = frame.getHeader()
    tai = hdr.get('TAI')
    stripe = hdr.get('STRIPE')
    strip = hdr.get('STRIP')
    obj = hdr.get('OBJECT')
    info.update(tai=tai, stripe=stripe, strip=strip, object=obj)

    astrans = frame.getAsTrans()
    wcs = SdssWcs(astrans)
    #print 'Created SDSS Wcs:', wcs
    #print '(x,y) = 1,1 -> RA,Dec', wcs.pixelToPosition(1,1)

    if roiradecsize is not None:
        ra,dec,S = roiradecsize
        fxc,fyc = wcs.positionToPixel(RaDecPos(ra,dec))
        print 'ROI center RA,Dec (%.3f, %.3f) -> x,y (%.2f, %.2f)' % (ra, dec, fxc, fyc)
        xc,yc = [int(np.round(p)) for p in fxc,fyc]

        roi = [np.clip(xc-S, 0, W),
               np.clip(xc+S, 0, W),
               np.clip(yc-S, 0, H),
               np.clip(yc+S, 0, H)]
        roi = [int(x) for x in roi]
        if roi[0]==roi[1] or roi[2]==roi[3]:
            print "ZERO ROI?", roi
            print 'S = ', S, 'xc,yc = ', xc,yc
            #assert(False)
            return None,None

        #print 'roi', roi
        #roi = [max(0, xc-S), min(W, xc+S), max(0, yc-S), min(H, yc+S)]
        info.update(roi=roi)

    if roiradecbox is not None:
        ra0,ra1,dec0,dec1 = roiradecbox
        xy = []
        for r,d in [(ra0,dec0),(ra1,dec0),(ra0,dec1),(ra1,dec1)]:
            xy.append(wcs.positionToPixel(RaDecPos(r,d)))
        xy = np.array(xy)
        xy = np.round(xy).astype(int)
        x0 = xy[:,0].min()
        x1 = xy[:,0].max()
        y0 = xy[:,1].min()
        y1 = xy[:,1].max()
        #print 'ROI box RA (%.3f,%.3f), Dec (%.3f,%.3f) -> xy x (%i,%i), y (%i,%i)' % (ra0,ra1, dec0,dec1, x0,x1, y0,y1)
        roi = [np.clip(x0,   0, W),
               np.clip(x1+1, 0, W),
               np.clip(y0,   0, H),
               np.clip(y1+1, 0, H)]
        #print 'ROI xy box clipped x [%i,%i), y [%i,%i)' % tuple(roi)
        if roi[0] == roi[1] or roi[2] == roi[3]:
            #print 'Empty roi'
            return None,None
        info.update(roi=roi)

        
    if roi is not None:
        x0,x1,y0,y1 = roi
    else:
        x0 = y0 = 0
    # Mysterious half-pixel shift.  asTrans pixel coordinates?
    wcs.setX0Y0(x0 + 0.5, y0 + 0.5)

    #print 'Band name:', bandname

    if nanomaggies:
        photocal = LinearPhotoCal(1., band=bandname)
    else:
        photocal = SdssNanomaggiesPhotoCal(bandname)

    sky = 0.
    skyobj = ConstantSky(sky)

    calibvec = frame.getCalibVec()

    invvarAtCenter = invvarAtCenter or invvarAtCenterImage

    psfield = sdss.readPsField(run, camcol, field)
    iva = dict(ignoreSourceFlux=invvarIgnoresSourceFlux)
    if invvarAtCenter:
        if roi:
            iva.update(constantSkyAt=((x0+x1)/2., (y0+y1)/2.))
        else:
            iva.update(constantSkyAt=(W/2., H/2.))
    invvar = frame.getInvvar(psfield, bandnum, **iva)
    invvar = invvar.astype(np.float32)
    if not invvarAtCenter:
        assert(invvar.shape == (H,W))

    # Could get this from photoField instead
    # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/photoField.html
    gain = psfield.getGain(bandnum)
    darkvar = psfield.getDarkVariance(bandnum)

    meansky = np.mean(frame.sky)
    meancalib = np.mean(calibvec)
    skysig = sqrt((meansky / gain) + darkvar) * meancalib

    # Added by @bpartridge to support photon counts
    # Calculate the Poisson-distributed number of electrons detected by the instrument
    dn = frame.getImage() / frame.getCalibVec() + frame.getSky()
    nelec = dn * gain

    info.update(sky=sky, skysig=skysig)
    zr = np.array(zrange)*skysig + sky
    info.update(zr=zr)

    # http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
    fpM = sdss.readFpM(run, camcol, field, bandname)

    if roi is None:
        image = frame.getImage()

    else:
        roislice = (slice(y0,y1), slice(x0,x1))
        image = frame.getImageSlice(roislice).astype(np.float32)

        # Added by @bpartridge to support photon counts
        dn = dn[roislice].copy()
        nelec = nelec[roislice].copy()

        if invvarAtCenterImage:
            invvar = invvar + np.zeros(image.shape, np.float32)
        elif invvarAtCenter:
            pass
        else:
            invvar = invvar[roislice].copy()

        H,W = image.shape

    # Added by @bpartridge to support photon counts
    info.update(dn=dn, nelec=nelec, nmgy=hdr.get('NMGY'))

    if (not invvarAtCenter) or invvarAtCenterImage:
        for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
            fpM.setMaskedPixels(plane, invvar, 0, roi=roi)

    if psf == 'kl-pix':
        # Pixelized KL-PSF
        klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
        # Trim symmetric zeros
        sh,sw = klpsf.shape
        while True:
            if (np.all(klpsf[0,:] == 0.) and
                np.all(klpsf[:,0] == 0.) and
                np.all(klpsf[-1,:] == 0.) and
                np.all(klpsf[:,-1] == 0.)):
                klpsf = klpsf[1:-1, 1:-1]
            else:
                break

        mypsf = PixelizedPSF(klpsf)
        
    elif psf == 'kl-gm':
        from tractor.emfit import em_fit_2d
        from tractor.fitpsf import em_init_params
        
        # Create Gaussian mixture model PSF approximation.
        klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
        S = klpsf.shape[0]
        # number of Gaussian components
        K = 3
        w,mu,sig = em_init_params(K, None, None, None)
        II = klpsf.copy()
        II /= II.sum()
        # HIDEOUS HACK
        II = np.maximum(II, 0)
        #print 'Multi-Gaussian PSF fit...'
        xm,ym = -(S/2), -(S/2)
        if savepsfimg is not None:
            plt.clf()
            plt.imshow(II, interpolation='nearest', origin='lower')
            plt.title('PSF image to fit with EM')
            plt.savefig(savepsfimg)
        res = em_fit_2d(II, xm, ym, w, mu, sig)
        #print 'em_fit_2d result:', res
        if res == 0:
            # print 'w,mu,sig', w,mu,sig
            mypsf = GaussianMixturePSF(w, mu, sig)
            mypsf.computeRadius()
        else:
            # Failed!  Return 'dg' model instead?
            print 'PSF model fit', psf, 'failed!  Returning DG model instead'
            psf = 'dg'
    if psf == 'dg':
        dgpsf = psfield.getDoubleGaussian(bandnum)
        print 'Creating double-Gaussian PSF approximation'
        (a,s1, b,s2) = dgpsf
        mypsf = NCircularGaussianPSF([s1, s2], [a, b])

    if brightpsf:
        print 'Wrapping PSF in SdssBrightPSF'
        (a1,s1, a2,s2, a3,sigmap,beta) = psfield.getPowerLaw(bandnum)
        mypsf = SdssBrightPSF(mypsf, a1,s1,a2,s2,a3,sigmap,beta)
        print 'PSF:', mypsf

    timg = Image(data=image, invvar=invvar, psf=mypsf, wcs=wcs,
                 sky=skyobj, photocal=photocal,
                 name=('SDSS (r/c/f/b=%i/%i/%i/%s)' %
                       (run, camcol, field, bandname)),
                 time=TAITime(tai),
                 **imargs)
    timg.zr = zr

    all_data = dict()
    all_data['img'] = timg
    all_data['counts'] = dn
    all_data['info'] = info

    filename = 'real_data/data_%d_%d_%d_%s.pkl' % (run, camcol, field, bandname)
    output = open(filename, 'wb')    
    pickle.dump(all_data, output)

    return timg,info



#
# Code to put together URLs for DR7 fields
#
def get_sdss_dr7_frame_url(run, camcol, field, band, rerun=40):
    # url to "corrected" frame 
    url_base  = "http://das.sdss.org/imaging/{run}/{rerun}/corr/{camcol}/" . \
                    format(rerun=rerun, run=run, camcol=camcol)
    file_name = "fpC-{run}-{band}{camcol}-{field}.fit.gz" . \
                    format(run="%06d"%run, band=band, camcol=camcol, field="%04d"%field)
    url = os.path.join(url_base, file_name)

    # psField url
    psurl_base = "http://das.sdss.org/imaging/{run}/{rerun}/objcs/{camcol}/" . \
                      format(rerun=rerun, run=run, camcol=camcol)
    psfield     = "psField-{run}-{camcol}-{field}.fit".format(run="%06d"%run, camcol=camcol, field="%04d"%field)
    psfield_url = os.path.join(psurl_base, psfield)

    # calibration statistics url
    calib_base = "http://das.sdss.org/imaging/{run}/{rerun}/calibChunks/{camcol}/" .\
                        format(rerun=rerun, run=run, camcol=camcol)
    calib      = "tsField-{run}-{camcol}-{rerun}-{field}.fit".format(run="%06d"%run, camcol=camcol, field="%04d"%field, rerun=rerun)
    calib_url  = os.path.join(calib_base, calib)
    return url, psfield_url, calib_url

def get_sdss_dr7_frame(run, camcol, field, band, rerun=40):
    url, psfield_url, calib_url = \
        get_sdss_dr7_frame_url(run, camcol, field, band, rerun)

    # download files if necessary
    import wget
    if not os.path.exists(img_filename):
        img_filename = wget.download(url)
    if not os.path.exists(ps_filename):
        ps_filename  = wget.download(psfield_url)

    # load calibration data to get sky noise
    ps_data  = fitsio.FITS(ps_filename)[6].read()

    # create fitsfile
    img_data = fitsio.FITS(img_filename)[0].read()
    img_header = fitsio.read_header(img_filename)

    import CelestePy.fits_image as fits_image
    reload(fits_image)
    imgfits = fits_image.FitsImage(band,
                              timg=imgs[band],
                              calib=1.,
                              gain=gain,
                              darkvar=darkvar,
                              sky=0.)


