import astrometry.util.fits as aufits
import tractor.sdss as sdss
import astrometry.sdss as asdss
from CelestePy.util.bound.bounding_box import get_bounding_boxes_idx
from CelestePy.util.data import make_fits_images, tractor_src_to_celestepy_src
import numpy as np
import matplotlib.pyplot as plt

from CelestePy.celeste import FitsImage
from CelestePy.celeste_galaxy_conditionals import gen_galaxy_psf_image

BANDS = ['u', 'g', 'r', 'i', 'z']

############################################
# TODO: factor this out into celeste.py -  #
############################################
#def gen_point_source_psf_image_with_fluxes(src_params, fits_image, return_patch=True, psf_grid=None):
#    src_img, ylim, xlim  = gen_point_source_psf_image(src_params.u, fits_image, return_patch=True, psf_grid=psf_grid)
#    flux     = src_params.fluxes[BANDS.index(fits_image.band)]
#    src_img *= (flux / fits_image.calib) * fits_image.kappa
#    return src_img, ylim, xlim
#
#def gen_celeste_image(src_params, fits_image):
#    if src_params.a == 0:
#        src_img, ylim, xlim = gen_point_source_psf_image_with_fluxes(src_params, fits_image)
#    else:
#        src_img, ylim, xlim = gen_galaxy_psf_image(
#            th = [src_params.theta, src_params.sigma, src_params.phi, src_params.rho],
#            u_s = src_params.u,
#            img = fits_image)
#    return src_img, ylim, xlim


def compare_small_patch(tractor_srcs, imgfits):
    """reads in a known image and source and compares our model 
    generation to the tractor's """

    # determine brightest sources - look at those first
    rbrightnesses = np.array([src.getBrightnesses()[0][2] for src in tractor_srcs])
    bright_i      = np.argsort(rbrightnesses)

    # grab one of the sources, plot celeste rendering of it and some statistics
    i           = bright_i[10]
    src         = tractor_srcs[i]
    celeste_src = tractor_src_to_celestepy_src(src)
    print "    visualizing source %d : "%i
    print "      tractor params      : ", src
    print "      celeste params      : ", src_params

    # plot the CelestePy model image with Tractor Parameters as a sanity check
    BANDS_TO_PLOT = ['r', 'i']
    fig, axarr = plt.subplots(len(BANDS_TO_PLOT), 4)
    for bi, b in enumerate(BANDS_TO_PLOT):

        # generate a celeste image patch
        src_img, ylim, xlim = gen_celeste_image(celeste_src, imgfits[b])

        # grab corresponding data patch - subtract out sky noise
        dpatch = imgfits[b].nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        dpatch -= np.median(dpatch)

        # plot the image
        axarr[bi,0].imshow(dpatch)
        axarr[bi,0].set_title("%s band (data)")
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
    plt.savefigure("patch_comparison.pdf", bbox_inches='tight')
    plt.close("all")


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
                f_s, ylim, xlim = gen_galaxy_psf_image(
                th = [src_params.theta, src_params.sigma, src_params.phi, src_params.rho],
                u_s = src_params.u,
                img = imgfits[b])

            modelims[band][ylim[0]:ylim[1], xlim[0]:xlim[1]] += f_s * src_params.fluxes[j]

    return modelims


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

    ##############################################
    # load in a full field and tractor sources   #
    #############################################
    run, camcol, field = 125, 1, 17
    tsrcs              = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgfits            = make_fits_images(run, camcol, field)
    imgs               = [imgfits[b] for b in BANDS]
    srcs               = [tractor_src_to_celestepy_src(s) for s in tsrcs]

    # stack into array - for comparison
    bands = ['u', 'g', 'r', 'i', 'z']
    flux_array = np.array([s.fluxes for s in srcs])
    tractor_fluxes = pd.DataFrame(flux_array, columns=bands)

    #############################################
    # initialize celeste model
    #############################################
    import CelestePy.models as models
    reload(models)
    model = models.Celeste(
            star_flux_prior_distn = None,
            gal_flux_prior_distn  = None,
            # patch epsilon options 
            )

    # for each run/camcol/field, add a data
    model.add_field(img_dict = imgfits)
    model.initialize_sources(init_src_params=srcs)

    # do a gibbs step
    model.field_list[0].resample_photons(model.srcs)
    model.srcs[0].resample()
    model.resample_model()



    # debug plot code
    #compare_small_patch(None, imgs)
    #raise "noooo"


        ###### GIBBS SAMPLE SOURCE PHOTONS ##########
    ##
    ## 1.  for each image, sample the source specific counts (Z_{n,m,s})
    ##
    img = imgs[2]
    img.epsilon = np.median(img.nelec)

    #samp_imgs, noise_sum = \
    #    sample_source_photons_single_image(img, srcs)
    from CelestePy.celeste_mcmc import \
        sample_source_photons_single_image_cython
    samp_imgs, noise_sum = \
        sample_source_photons_single_image_cython(img, srcs)

    # time it!
    #%timeit -r 1 sample_source_photons_single_image_cython(img, srcs)
    #samp_imgs, noise_sum = sample_source_photons_single_image(img, srcs)

    ###### DEBUG PLOT ##################
    # plot a few samples
    src_types     = np.array([type(src) != PointSource for src in tsrcs])
    rbrightnesses = np.array([src.getBrightnesses()[0][2] for src in tsrcs])
    bright_i      = np.argsort(rbrightnesses)
    fig, axarr = plt.subplots(3, 3)
    src_idxs = bright_i[:9]
    src_idxs = np.where(src_types)[0][:9]
    for idx, ax in zip(src_idxs, axarr.flatten()):
        ax.imshow(np.asarray(samp_imgs[idx].data), interpolation='none')
    plt.show()


    ## resample image noise given poisson nums
    # eps ~ noise_count

    ## for each source, resample images

