print "importing matplotlib"
import matplotlib.pyplot     as plt
print "done importing matplotlib"

import autograd.numpy.linalg as npla
import autograd.numpy        as np
import autograd.numpy.random as npr
import autograd.scipy.misc   as scpm
from autograd import grad

import tractor.sdss as sdss
import astrometry.sdss as asdss
import astrometry.util.fits as aufits

from scipy.stats.distributions import gamma
import CelestePy.celeste as celeste
import CelestePy.celeste_galaxy_conditionals as galaxies
from CelestePy.util.data.get_data import tractor_src_to_celestepy_src
from CelestePy.celeste_src import SrcParams

from scipy.optimize import minimize

############################################################################
# Likelihoods of varying shapes/dimensionality for testing samplers
############################################################################

BANDS = ['u', 'g', 'r', 'i', 'z']

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

        imgfits[band] = celeste.FitsImage(iband,
                                          timg=imgs[band],
                                          calib=calib,
                                          gain=gain,
                                          darkvar=darkvar,
                                          sky=sky)
    return imgfits


from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def add_colorbar_to_axis(ax, cim):
    """ pretty generic helper function to throw a colorbar onto an axis """
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="10%", pad=0.05)
    cbar    = plt.colorbar(cim, cax=cax)
    # Manually set ticklabels (not ticklocations, they remain unchanged)
    #ax4.set_yticklabels([0, 50, 30, 'foo', 'bar', 'baz'])


if __name__=="__main__":

    # galaxy parameters: loc, shape, etc in arg here
    # TODO: actually make this reflect how 
    #       galaxy location, shape parameters actually create gaussian 
    #       mixture parameters

    def create_pixel_grid(image, loc):
        v_s = image.equa2pixel(loc)
        bound = image.R
        minx_b, maxx_b = max(0, int(v_s[0] - bound)), min(int(v_s[0] + bound + 1), image.nelec.shape[1])
        miny_b, maxy_b = max(0, int(v_s[1] - bound)), min(int(v_s[1] + bound + 1), image.nelec.shape[0])
        y_grid = np.arange(miny_b, maxy_b, dtype=np.float)
        x_grid = np.arange(minx_b, maxx_b, dtype=np.float)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
        pixel_grid = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))

        return xx.astype(int), yy.astype(int),pixel_grid

    def gen_galaxy_image(pixel_info, images, fluxes, loc,
                         gal_theta, gal_sig, gal_rho, gal_phi):
        xx = pixel_info[0]
        pixel_grid = pixel_info[2]
        bandims = np.zeros((xx.shape[0], xx.shape[1], len(images)))
        for idx,image in enumerate(images):
            im = gen_galaxy_psf_image(pixel_info[2], image, loc,
                                      gal_theta, gal_sig, gal_rho, gal_phi,
                                      image.weights, image.means, image.covars)
            bandims[:,:,idx] = fluxes[idx] * im.reshape(xx.shape, order='C')

        return bandims

    def gen_point_source_image(pixel_info, images, fluxes, loc):
        xx = pixel_info[0]
        pixel_grid = pixel_info[2]
        bandims = np.zeros((xx.shape[0], xx.shape[1], len(images)))
        for idx,image in enumerate(images):
            im = gen_point_source_psf_image(pixel_grid, image, loc,
                                            image.weights, image.means, image.covars)
            bandims[:,:,idx] = fluxes[idx] * im.reshape(xx.shape, order='C')

        return bandims

    def calc_galaxy_prior():
        return 0 

    def calc_point_source_prior():
        return 0

    def calc_total_prob_galaxy(images, fluxes, loc, shape):
        xx,yy,pixel_grid = create_pixel_grid(images[0], loc)
        pixel_info = [xx, yy, pixel_grid]
        prior = calc_galaxy_prior()
        lams  = gen_galaxy_image(pixel_info, images, fluxes, loc,
                                 shape[0], shape[1], shape[2], shape[3])
        curr_sum = prior
        for idx,image in enumerate(images):
            curr_sum += np.sum(image.nelec[yy,xx] * np.log(lams[:,:,idx]) - lams[:,:,idx])


        # verify galaxy
        fig, axarr = plt.subplots(1, 2)
        axarr[0].contourf(xx, yy, lams[:,:,0])
        axarr[1].contourf(xx, yy, image.nelec[yy,xx])
        plt.show()

        return curr_sum, pixel_info

    def calc_total_prob_point_source(pixel_info, images, fluxes, loc):
        xx = pixel_info[0]
        yy = pixel_info[1]
        prior = calc_point_source_prior()
        lams  = gen_point_source_image(pixel_info, images, fluxes, loc)
        curr_sum = prior
        for idx,image in enumerate(images):
            curr_sum += np.sum(image.nelec[yy,xx] * np.log(lams[:,:,idx]) - lams[:,:,idx])

        return curr_sum

    NUM_BANDS = 5
    NUM_LOC = 2
    NUM_SHAPE = 4
    def squared_loss(galaxy_src, point_src, images):
        loss = 0
        for image in images:
            galaxy_im = celeste.gen_src_image(galaxy_src, image, return_patch=False)
            point_src_im = celeste.gen_src_image(point_src, image, return_patch=False)
            loss += np.sum(np.sum((galaxy_im - point_src_im)**2))

        return loss

    # read in image and corresponding source
    print "read in images and sources"
    run = 125
    camcol = 1
    field = 17
    tsrcs = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgfits = make_fits_images(run, camcol, field)

    # list of images, list of celeste sources
    imgs = [imgfits[b] for b in BANDS]
    srcs = [tractor_src_to_celestepy_src(s) for s in tsrcs]

    src_types = np.array([src.a for src in srcs])
    rfluxes = np.array([src.fluxes[2] for src in srcs])
    rfluxes[src_types == 0] = -1
    brightest_i = np.argmax(rfluxes)

    def star_arg_squared_loss(fluxes, galaxy_src, images):
        star = SrcParams(src.u, a=0, fluxes=np.exp(fluxes))
        return squared_loss(galaxy_src, star, images)

    # do gradient descent
    # 1, 9, 10 galaxies
    for src in [srcs[46]]:
        if src.a == 0:
            continue

        star = SrcParams(src.u, a=0, fluxes=src.fluxes)
        print "loss, galaxy with itself:", squared_loss(src, src, imgs)
        print "loss, galaxy with star:", squared_loss(src, star, imgs)

        res = minimize(star_arg_squared_loss, np.log(src.fluxes), args=(src, imgs), method='Nelder-Mead', options={'maxiter':100})
        print "fluxes:", src.fluxes, res.x
        print " opt result: ", res

        # show the new star
        fig, axarr  = plt.subplots(len(BANDS), 2)
        for bi, b in enumerate(BANDS):
            final_galaxy_im = celeste.gen_src_image(src, imgs[bi])

            final_star = SrcParams(src.u, a=0, fluxes=np.exp(res.x))
            final_star_im = celeste.gen_src_image(final_star, imgs[bi])

            gim = axarr[bi,0].imshow(final_galaxy_im)
            sim = axarr[bi,1].imshow(final_star_im)
            add_colorbar_to_axis(axarr[bi,0], gim)
            add_colorbar_to_axis(axarr[bi,1], sim)

            for c in range(2):
                axarr[bi,c].get_xaxis().set_visible(False)
                axarr[bi,c].get_yaxis().set_visible(False)

        axarr[0,0].set_title('galaxy patch')
        axarr[0,1].set_title('star patch')
        fig.tight_layout()
        plt.show()


