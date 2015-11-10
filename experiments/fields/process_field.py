import astrometry.util.fits as aufits
import tractor.sdss as sdss
from tractor.basics import PointSource
from tractor.galaxy import ExpGalaxy, DevGalaxy, CompositeGalaxy
from CelestePy import gen_point_source_psf_image, \
                      gen_galaxy_psf_image, FitsImage, SrcParams
import numpy as np
import matplotlib.pyplot as plt

BANDS = ['u', 'g', 'r', 'i', 'z']

def tractor_src_to_celestepy_src(tsrc):
    """Conversion between tractor source object and our source object...."""
    pos = tsrc.getPosition()
    u = [p for p in pos]

    # brightnesse are stored in mags (gotta convert to nanomaggies)
    def mags2nanomaggies(mags):
        return np.power(10., (mags - 22.5)/-2.5)
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
                         phi = shape[2] * np.pi / 180.,
                         sigma=shape[0],
                         rho=shape[1],
                         fluxes=fluxes)


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

    fn = sdss.DR9().retrieve('photoField', run, camcol, field)
    F = aufits.fits_table(fn)

    # convert to FitsImage's
    imgfits = {}
    for iband,band in enumerate(BANDS):
        print "converting images %s" % band
        frame   = sdss.DR9().readFrame(run, camcol, field, band)
        calib   = np.median(frame.getCalibVec())
        gain    = F[0].gain[iband]
        darkvar = F[0].dark_variance[iband]
        sky     = np.median(frame.getSky())
        imgfits[band] = FitsImage(band,
                                  timg=imgs[band],
                                  calib=calib,
                                  gain=gain,
                                  darkvar=darkvar,
                                  sky=sky)
    return imgfits

def gen_point_source_psf_image_with_fluxes(src_params, fits_image):
    src_img  = gen_point_source_psf_image(src_params.u, fits_image)
    flux     = src_params.fluxes[BANDS.index(fits_image.band)]
    src_img *= (flux / fits_image.calib) * fits_image.kappa
    return src_img

def compare_small_patch():
    """reads in a known image and source and compares our model 
    generation to the tractor's """
    run = 125
    camcol = 1
    field = 17

    # read in sources, images
    srcs = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgfits = make_fits_images(run, camcol, field)

    # track down the brightest sources in this field for sanity checking
    rbrightnesses = np.array([src.getBrightnesses()[0][2] for src in srcs])
    bright_i      = np.argsort(rbrightnesses)
    for i in bright_i[:50]:
        print srcs[i]

    i = bright_i[2]
    i = 1
    src = srcs[i]
    src_params = tractor_src_to_celestepy_src(src)

    # plot the CelestePy model image with Tractor Parameters as a sanity check
    BANDS_TO_PLOT = ['r', 'i']
    fig, axarr = plt.subplots(len(BANDS_TO_PLOT), 3)
    for bi, b in enumerate(BANDS_TO_PLOT):
        if src_params.a == 0:
            src_img = gen_point_source_psf_image_with_fluxes(src_params, imgfits[b])
        else:
            src_img = gen_galaxy_psf_image(src_params, imgfits[b]);

        pixel_loc = imgfits[b].equa2pixel(src_params.u)
        minx, maxx = pixel_loc[0] - 25, pixel_loc[0] + 25
        miny, maxy = pixel_loc[1] - 25, pixel_loc[1] + 25

        dpatch = imgfits[b].nelec[miny:maxy, minx:maxx]
        dpatch -= np.median(dpatch)
        mpatch = src_img[miny:maxy, minx:maxx]

        # check how good bounding box is
        bound = imgfits[b].R
        minx_b, maxx_b = pixel_loc[0] - bound, pixel_loc[0] + bound
        miny_b, maxy_b = pixel_loc[1] - bound, pixel_loc[1] + bound

        total_pixels = np.sum(np.sum(src_img))
        bounded_pixels = np.sum(np.sum(src_img[miny_b:maxy_b, minx_b:maxx_b]))
        percent_diff = np.abs(total_pixels - bounded_pixels) / total_pixels * 100
        print "Total pixels:", total_pixels
        print "Bounded pixels:", bounded_pixels
        print "% off:", percent_diff

        axarr[bi,0].imshow(dpatch)
        axarr[bi,1].imshow(mpatch)
        dim = axarr[bi,2].imshow( (dpatch - mpatch) )
        axarr[bi,2].set_title('diff (mean = %2.2f)'%np.mean(dpatch-mpatch))

        # remove x/y ticks
        for c in range(3):
            axarr[bi,c].get_xaxis().set_visible(False)
            axarr[bi,c].get_yaxis().set_visible(False)

    axarr[0,0].set_title('data patch')
    axarr[0,1].set_title('model patch')
    fig.tight_layout()
    plt.show()

def main(run, camcol, field):
    # read in sources, images
    srcs = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgfits = make_fits_images(run, camcol, field)

    # make model images
    modelims = {}
    for i,src in enumerate(srcs[0:50]):
        print "Source %d" % i
        # convert to Celeste sources
        src_params = tractor_src_to_celestepy_src(src)

        for j,band in enumerate(BANDS):
            if src_params.a == 0:
                f_s = gen_point_source_psf_image_with_fluxes(src_params, imgfits[band])
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

