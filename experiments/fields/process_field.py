import astrometry.util.fits as aufits
import tractor.sdss as sdss
from tractor.basics import PointSource
from tractor.galaxy import ExpGalaxy, DevGalaxy, CompositeGalaxy
from CelestePy import gen_point_source_psf_image, \
                      gen_galaxy_psf_image, FitsImage, SrcParams
import numpy as np
import matplotlib.pyplot as plt

def tsrc_to_src_params(tsrc):
    pos = tsrc.getPosition()
    u = [p for p in pos]
    fluxes = tsrc.getBrightnesses()[0]
    fluxes = [mags2nanomaggies(flux) for flux in fluxes]

    if type(tsrc) == PointSource:
        return SrcParams(u, a=0, fluxes=fluxes)
    else:
        if type(tsrc) == ExpGalaxy:
            shape = tsrc.getShape()
            theta = 0.
        elif type(tsrc) == DevGalaxy:
            shape = tsrc.getShape()
            theta = 1.
        elif type(tsrc) == CompositeGalaxy:
            shape = tsrc.shapeExp
            theta = 0.5
        else:
            pass

        return SrcParams(u,
                         a=1,
                         v=u,
                         theta=theta,
                         phi=shape[2],
                         sigma=shape[0],
                         rho=shape[1],
                         fluxes=fluxes)

def mags2nanomaggies(mags): 
    return np.power(10., (mags - 22.5)/-2.5)

def compare_small_patch():
    """ reads in a known image and source and compares our model 
    generation to the tractor's """
    run = 125
    camcol = 1
    field = 17

    # read in sources, images
    srcs = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgs = {}
    for band in BANDS:
        print "reading in band %s" % band
        imgs[band] = sdss.get_tractor_image_dr9(run, camcol, field, band)

    fn = sdss.DR9().retrieve('photoField', run, camcol, field)
    F = aufits.fits_table(fn)

    # convert to FitsImage's
    imgfits = {}
    for iband,band in enumerate(BANDS):
        print "converting images %s" % band
        frame   = sdss.DR9().readFrame(run, camcol, field, band)
        calib   = np.median(frame.getCalibVec())
        gain    = F.gain[0][iband]
        darkvar = F.dark_variance[iband]
        imgfits[band] = FitsImage(band,
                                  timg=imgs[band],
                                  calib=calib,
                                  gain=gain,
                                  darkvar=darkvar)

    ########################
    # debug               ##
    ########################
    # the first source is located within the xpixel=[690,740], ypixel=[150,200]
    # bounding box
    src = srcs[0]
    src_params = tsrc_to_src_params(src)
    src_img    = gen_point_source_psf_image(src_params.u, imgfits['r'])

    fig, axarr = plt.subplots(1, 3)
    dpatch = imgfits['r'].nelec[150:200, 690:740]
    mpatch = src_img[150:200, 690:740]
    axarr[0].imshow(dpatch)
    axarr[1].imshow(mpatch)
    dim = axarr[2].imshow( (dpatch - mpatch) / mpatch )
    axarr[0].set_title('data patch')
    axarr[1].set_title('model patch')
    axarr[2].set_title('diff (mean = %2.2f)'%np.sum((dpatch-mpatch)))
    plt.show()

BANDS = ['u', 'g', 'r', 'i', 'z']
def main(run, camcol, field):
    # read in sources, images
    srcs = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgs = {}
    for band in BANDS:
        print "reading in band %s" % band
        imgs[band] = sdss.get_tractor_image_dr9(run, camcol, field, band)

    fn = sdss.DR9().retrieve('photoField', run, camcol, field)
    F = aufits.fits_table(fn)

    # convert to FitsImage's
    imgfits = {}
    for iband,band in enumerate(BANDS):
        print "converting images %s" % band
        frame   = sdss.DR9().readFrame(run, camcol, field, band)
        calib   = np.median(frame.getCalibVec())
        gain    = F.gain[0][iband]
        darkvar = F.dark_variance[iband]

        imgfits[band] = FitsImage(band,
                                  timg=imgs[band],
                                  calib=calib,
                                  gain=gain,
                                  darkvar=darkvar)

    # get images
    modelims = {}
    for i,src in enumerate(srcs[0:50]):
        print "Source %d" % i
        # convert to Celeste sources
        src_params = tsrc_to_src_params(src)

        for j,band in enumerate(BANDS):
            if src_params.a == 0:
                f_s = gen_point_source_psf_image(src_params.u, imgfits[band])
            elif src_params.a == 1:
                f_s = gen_galaxy_psf_image(src_params, imgfits[band]);

            if band in modelims:
                modelims[band] += f_s * src_params.fluxes[j]
            else:
                modelims[band] = f_s * src_params.fluxes[j]

    return modelims

if __name__ == '__main__':
    run = 125
    camcol = 1
    field = 17
    #modelims = main(run, camcol, field)

    compare_small_patch()

