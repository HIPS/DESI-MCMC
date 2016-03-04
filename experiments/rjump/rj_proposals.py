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

        imgfits[band] = celeste.FitsImage(band,
                                          timg=imgs[band],
                                          calib=calib,
                                          gain=gain,
                                          darkvar=darkvar,
                                          sky=sky)
    return imgfits

def mog_like(x, means, icovs, dets, pis):
    """ compute the log likelihood according to a mixture of gaussians
        with means = [mu0, mu1, ... muk]
             icovs = [C0^-1, ..., CK^-1]
             dets = [|C0|, ..., |CK|]
             pis  = [pi1, ..., piK] (sum to 1)
        at locations given by x = [x1, ..., xN]
    """
    xx = np.atleast_2d(x)
    centered = xx[:,:,np.newaxis] - means.T[np.newaxis,:,:]
    solved   = np.einsum('ijk,lji->lki', icovs, centered)
    logprobs = -0.5*np.sum(solved * centered, axis=1) - np.log(2*np.pi) - 0.5*np.log(dets) + np.log(pis)
    logprob  = scpm.logsumexp(logprobs, axis=1)
    if len(x.shape) == 1:
        return np.exp(logprob[0])
    else:
        return np.exp(logprob)

if __name__=="__main__":

    # galaxy parameters: loc, shape, etc in arg here
    # TODO: actually make this reflect how 
    #       galaxy location, shape parameters actually create gaussian 
    #       mixture parameters


    def gen_prof_mog_params(image, loc,
                            gal_sig, gal_rho, gal_phi,
                            psf_weights, psf_means, psf_covars,
                            prof_amp, prof_sig):

        v_s = image.equa2pixel(loc)
        R = galaxies.gen_galaxy_transformation(gal_sig, gal_rho, gal_phi)
        W = np.dot(R, R.T)
        
        K_psf  = psf_weights.shape[0]
        K_prof = prof_amp.shape[0]

        # compute MOG components
        num_components = K_psf * K_prof
        weights = np.zeros(num_components, dtype=np.float)
        means   = np.zeros((num_components, 2), dtype=np.float)
        covars  = np.zeros((num_components, 2, 2), dtype=np.float)
        cnt     = 0
        for k in range(K_psf):                              # num PSF Componenets
            for j in range(K_prof):                         # galaxy type components
                ## compute weights and component mean/variances
                weights[cnt] = psf_weights[k] * prof_amp[j]

                ## compute means
                means[cnt,0] = v_s[0] + psf_means[k, 0]
                means[cnt,1] = v_s[1] + psf_means[k, 1]

                ## compute covariance matrices
                for ii in range(2):
                    for jj in range(2):
                        covars[cnt, ii, jj] = psf_covars[k, ii, jj] + \
                                              prof_sig[j] * W[ii, jj]

                # increment index
                cnt += 1

        icovs = np.array([npla.inv(c) for c in covars])
        dets  = np.array([npla.det(c) for c in covars])
        chols = np.array([npla.cholesky(c) for c in covars])
        return means, covars, icovs, dets, chols, weights

    def gen_galaxy_psf_image(pixel_grid, image, loc,
                             gal_theta, gal_sig, gal_rho, gal_phi,
                             psf_weights, psf_means, psf_covars):

        # generate MoG params
        gal_exp_amp = galaxies.galaxy_prof_dict['exp'].amp
        gal_dev_amp = galaxies.galaxy_prof_dict['dev'].amp
        gal_exp_sig = galaxies.galaxy_prof_dict['exp'].var[:,0,0]
        gal_dev_sig = galaxies.galaxy_prof_dict['dev'].var[:,0,0]

        means, covs, icovs, dets, chols, pis = \
                        gen_prof_mog_params(
                            image, loc,
                            gal_sig, gal_rho, gal_phi,
                            psf_weights, psf_means, psf_covars,
                            gal_exp_amp, gal_exp_sig)
 
        exp_like = mog_like(pixel_grid, means, icovs, dets, pis)

        means, covs, icovs, dets, chols, pis = \
                        gen_prof_mog_params(
                            image, loc,
                            gal_sig, gal_rho, gal_phi,
                            psf_weights, psf_means, psf_covars,
                            gal_dev_amp, gal_dev_sig)

        dev_like = mog_like(pixel_grid, means, icovs, dets, pis)
        
        # compute mog likelihod
        return gal_theta * exp_like + (1 - gal_theta) * dev_like

    def gen_point_source_psf_image(pixel_grid, image, loc,
                                   psf_weights, psf_means, psf_covars):
        # use image PSF
        icovs = np.array([npla.inv(c) for c in psf_covars])
        dets  = np.array([npla.det(c) for c in psf_covars])
        chols = np.array([npla.cholesky(c) for c in psf_covars])

        return mog_like(pixel_grid, psf_means, icovs, dets, psf_weights)

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
        axarr[0].contourf(xx, yy, lams[:,:,2])
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
    def loss(th0, th1, images):
        l1, pixel_grid = calc_total_prob_galaxy(images, th0[:NUM_BANDS],
                            th0[NUM_BANDS:NUM_BANDS + NUM_LOC],
                            th0[NUM_BANDS + NUM_LOC:NUM_BANDS + NUM_LOC + NUM_SHAPE])
        l2 = calc_total_prob_point_source(pixel_grid, images,
                                          th0[:NUM_BANDS],
                                          th0[NUM_BANDS:NUM_BANDS + NUM_LOC])
        return (l1 - l2) * (l1 - l2)

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

    # do gradient descent
    # 1, 9, 10 galaxies
    print [src.a for src in srcs]
    for src in [srcs[10]]:
        if src.a == 0:
            continue

        print "looking at source", src
        shape_params = np.array([src.theta, src.phi, src.sigma, src.rho])
        galaxy_params = np.concatenate((src.fluxes, src.u, shape_params))
        print "galaxy params", galaxy_params

        star_params = np.concatenate((src.fluxes, src.u))
        print "initial star params", star_params
        alpha = 0.01
        loss(galaxy_params, star_params, imgs)
        break
        #while True:
        #    star_params -= alpha * grad(loss, argnum=1)(galaxy_params, star_params, imgs)


"""
    fig, axarr = plt.subplots(1, 2)
    axarr[0].contourf(xx, yy, np.exp(M1).reshape(xx.shape))
    axarr[1].contourf(xx, yy, np.exp(M2).reshape(xx.shape))
    plt.show()
"""

