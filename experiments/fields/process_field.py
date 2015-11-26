import astrometry.util.fits as aufits
import tractor.sdss as sdss
import astrometry.sdss as asdss
from tractor.basics import PointSource
from tractor.galaxy import ExpGalaxy, DevGalaxy, CompositeGalaxy
from CelestePy import gen_point_source_psf_image, \
                      gen_galaxy_psf_image, FitsImage, SrcParams, gen_psf_src_image_bound
from CelestePy.util.bound.bounding_box import get_bounding_boxes_idx
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

    fn = asdss.DR9().retrieve('photoField', run, camcol, field)
    F = aufits.fits_table(fn)

    # convert to FitsImage's
    imgfits = {}
    for iband,band in enumerate(BANDS):
        print "converting images %s" % band
        frame   = asdss.DR9().readFrame(run, camcol, field, band)
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

def gen_point_source_psf_image_with_fluxes(src_params, fits_image, return_patch=True, psf_grid=None):
    src_img, ylim, xlim  = gen_point_source_psf_image(src_params.u, fits_image, return_patch=True, psf_grid=psf_grid)
    flux     = src_params.fluxes[BANDS.index(fits_image.band)]
    src_img *= (flux / fits_image.calib) * fits_image.kappa
    return src_img, ylim, xlim




def compare_small_patch(src_params, imgfits):
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

    i = bright_i[31]
    src = srcs[i]
    src_params = tractor_src_to_celestepy_src(src)
    print "New source:", src_params

    # plot the CelestePy model image with Tractor Parameters as a sanity check
    BANDS_TO_PLOT = ['r', 'i']
    fig, axarr = plt.subplots(len(BANDS_TO_PLOT), 3)
    for bi, b in enumerate(BANDS_TO_PLOT):
        if src_params.a == 0:
            src_img, ylim, xlim = gen_point_source_psf_image_with_fluxes(src_params, imgfits[b])
        else:
            src_img, ylim, xlim = gen_galaxy_psf_image(src_params, imgfits[b]);

        # grab patches
        dpatch = imgfits[b].nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        dpatch -= np.median(dpatch)
        mpatch = src_img

        #pixel_loc = imgfits[b].equa2pixel(src_params.u)
        #minx, maxx = pixel_loc[0] - 25, pixel_loc[0] + 25
        #miny, maxy = pixel_loc[1] - 25, pixel_loc[1] + 25

        #dpatch = imgfits[b].nelec[miny:maxy, minx:maxx]
        #dpatch -= np.median(dpatch)
        #mpatch = src_img[miny:maxy, minx:maxx]

        # check how good bounding box is
        #bound = imgfits[b].R
        #minx_b, maxx_b = pixel_loc[0] - bound, pixel_loc[0] + bound
        #miny_b, maxy_b = pixel_loc[1] - bound, pixel_loc[1] + bound

        #total_pixels = np.sum(np.sum(src_img))
        #bounded_pixels = np.sum(np.sum(src_img[miny_b:maxy_b, minx_b:maxx_b]))
        #percent_diff = np.abs(total_pixels - bounded_pixels) / total_pixels * 100
        #print "Total pixels:", total_pixels
        #print "Bounded pixels:", bounded_pixels
        #print "% off:", percent_diff

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


#####################
# Render full field #
#####################
def main(imgfits, srcs):
    modelims = {}
    modelims = {b: np.zeros(imgfits[b].nelec.shape, dtype=np.float) for b in BANDS}
    for i,src in enumerate(srcs):
        if i % 50 == 0:
            print "src %d of %d"%(i, len(srcs))
        # convert to Celeste sources
        src_params = tractor_src_to_celestepy_src(src)

        for j,band in enumerate(BANDS):
            if src_params.a == 0:
                f_s, ylim, xlim = gen_point_source_psf_image_with_fluxes(src_params, imgfits[band], return_patch=True)
            elif src_params.a == 1:
                f_s, ylim, xlim = gen_galaxy_psf_image(src_params, imgfits[band]);

            modelims[band][ylim[0]:ylim[1], xlim[0]:xlim[1]] += f_s * src_params.fluxes[j]

    return modelims


def gen_src_image_with_fluxes(src, img):
    if src.a == 0:
        f_s, ylim, xlim = gen_point_source_psf_image_with_fluxes(src, img)
    elif src.a == 1:
        psf_img, ylim, xlim = gen_galaxy_psf_image(src, img)
        gal_flux = (src.fluxes[BANDS.index(img.band)] / img.calib ) * img.kappa
        f_s      = gal_flux * psf_img
    return f_s, ylim, xlim


def sample_source_photons_single_image(img, srcs):

    # compute source boxes
    src_locs  = np.row_stack([img.equa2pixel(s.u) for s in srcs])
    imgR      = np.array([gen_psf_src_image_bound(s, img) for s in srcs])
    src_boxes = np.column_stack([
                    np.floor(src_locs[:,0] - imgR),
                    np.ceil(src_locs[:,0] + imgR),
                    np.floor(src_locs[:,1] - imgR),
                    np.ceil(src_locs[:,1] + imgR)
                    ])

    # generate model image for each source
    src_imgs = [gen_src_image_with_fluxes(s, img) for s in srcs]

    # sampled images
    samp_imgs = [(np.zeros(s.shape), ylim, xlim) for s, ylim, xlim in src_imgs]

    # keep track of noise sum in image (for inferring img.epsilon)
    noise_sum = 0.

    cnt = 0
    for (y, x), num_photons_xy in np.ndenumerate(img.nelec):
        possible_srcs = get_bounding_boxes_idx(np.array([x,y]), src_boxes)
        if len(possible_srcs) == 0:
            noise_sum += num_photons_xy
            continue

        # if only one possible source, don't sample from multinomial...
        #if len(possible_srcs) == 1:
        #    src_photons = np.array([num_photons_xy])
        #else:
        model_fluxes = np.zeros(len(possible_srcs)+1)
        for i, pi in enumerate(possible_srcs):
            imgpatch, ylim, xlim = src_imgs[pi]
            if (y-ylim[0] >= 0) and (y-ylim[0] < imgpatch.shape[0]) and \
                    (x-xlim[0] >= 0) and (x-xlim[0] < imgpatch.shape[1]):
                model_fluxes[i] = imgpatch[y-ylim[0], x-xlim[0]]

        # tack on sky noise on the end
        model_fluxes[-1] = img.epsilon
        #if np.all(model_fluxes == 0.):
        #   continue

        # multinomial sample
        p = model_fluxes / model_fluxes.sum()
        src_photons = np.random.multinomial(int(num_photons_xy), p)

        # store in sampled images
        for i, pi in enumerate(possible_srcs):
            simg, ylim, xlim = samp_imgs[pi]
            if (y-ylim[0] >= 0) and (y-ylim[0] < simg.shape[0]) and \
                    (x-xlim[0] >= 0) and (x-xlim[0] < simg.shape[1]):
                simg[y-ylim[0], x-xlim[0]] = src_photons[i]

        # increment sky noise summary
        noise_sum += src_photons[-1]

        # verbose??
        cnt += 1
        if cnt % 10000== 0:
            print "%d of %d"%(cnt, np.prod(img.nelec.shape))

    return samp_imgs, noise_sum


if __name__ == '__main__':
    run = 125
    camcol = 1
    field = 17
    tsrcs = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgfits = make_fits_images(run, camcol, field)

    # list of images, list of celeste sources
    imgs = [imgfits[b] for b in BANDS]
    srcs = [tractor_src_to_celestepy_src(s) for s in tsrcs]

    ###### GIBBS SAMPLE SOURCE PHOTONS ##########
    ##
    ## 1.  for each image, sample the source specific counts (Z_{n,m,s})
    ##
    img = imgs[2]
    img.epsilon = np.median(img.nelec)
    samp_imgs, noise_sum = sample_source_photons_single_image(img, srcs)


    ## resample image noise given poisson nums
    # eps ~ noise_count

    ## for each source, resample images



    import sys
    sys.exit()


    modelims = main(imgfits, srcs) 

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(modelims['r'], interpolation='none')
    axarr[1].imshow(imgfits['r'].nelec, interpolation='none')
    plt.show()

    # read in sources, images

    ###################################################
    # compare small patch on a bright source          #
    ###################################################
    # track down the brightest sources in this field for sanity checking
    rbrightnesses = np.array([src.getBrightnesses()[0][2] for src in tsrcs])
    bright_i      = np.argsort(rbrightnesses)
    ##for i in bright_i[:50]:
    #    print srcs[i]
    i = bright_i[26]
    src = srcs[i]
    src_params = tractor_src_to_celestepy_src(src)

    # plot small patch
    compare_small_patch(src_params, imgfits)


    for bi in bright_i[:30]:
        print bi, tsrcs[bi]

############################
# Sample source photons
############################
#def sample_source_photons(imgs, srcs):
#
#    #src_boxes = np.array([
#
#    printif("    sampling Z's", verbose)
#    all_src_images = []
#    for img in imgs:
#        src_probs = gen_src_prob_layers(srcs, img)  # S x Npix x Mpix
#        src_image = np.zeros(src_probs.shape)
#        for (i,j), xij in np.ndenumerate(img.nelec):
#
#            # 
#            possible_sources = get_bounding_boxes_idx(pixel_loc, src_boxes)
#
#            src_image[:,i,j] = np.random.multinomial(int(xij), src_probs[:,i,j])
#        all_src_images.append(src_image)
#
#        #### debug #####
#        if False:
#            if img.band == 'r':
#                fig, axarr = plt.subplots(1, src_image.shape[0])
#                subplot_imshow_colorbar(src_image, fig, axarr)
#                plt.show()
#        #### end debug ####
#    printif("      .... done", verbose)
#
######### END SAMPLER ############


