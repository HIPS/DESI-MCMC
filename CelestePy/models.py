"""
CelestePy catalog of the universe.

Andrew Miller <acm@seas.harvard.edu>
"""

import autograd.numpy as np
import CelestePy.celeste_mcmc as cel_mcmc
import CelestePy.util.data as du
from  CelestePy import celeste
import CelestePy.celeste_galaxy_conditionals as gal_funs
from CelestePy.util.infer.slicesample import slicesample
import pyprind

class CelesteBase(object):
    """ Main model class - interface to user.  Holds a list of
    Source objects, each of which contains local markov chain state, 
    including source parameters, source-sample images"""
    def __init__(self, gal_flux_prior_distn=None, star_flux_prior_distn=None):
        #TODO incorporate priors over fluxes
        # model keeps track of a field list
        self.field_list = []
        self.bands = ['u', 'g', 'r', 'i', 'z']

        #TODO incorporate prior over fluxes
        self.star_flux_prior_distn = star_flux_prior_distn
        self.gal_flux_prior_distn  = gal_flux_prior_distn
        #self.gal_shape_prior_distn = gal_shape_prior_distn

    def initialize_sources(self, init_srcs=None, init_src_params=None, photoobj_df=None):
        """ initialize sources after adding fields """
        if init_srcs is not None:
            self.srcs = init_srcs
        elif init_src_params is not None:
            self.srcs = [self._source_type(s, self) for s in init_src_params]
        elif photoobj_df is not None:
            self.srcs = [self._source_type(du.photoobj_to_celestepy_src(p), self)
                         for (i, p) in photoobj_df.iterrows()]
        else:
            raise NotImplementedError

    def intialize_from_fields(self):
        """initialize sources from bright spots - make everything a point source?"""
        raise NotImplementedError

    def add_field(self, img_dict):
        """ add a field (run/camcol/field) image information to the model
                img_dict  = fits image keyed by 'ugriz' band
                init_srcs = sources (tractor or celeste) initialized in this field
        """
        for k in img_dict.keys():
            assert k in self.bands, "Celeste model doesn't support band %s"%k
        self.field_list.append(Field(img_dict))

    @property
    def source_types(self):
        return np.array([s.object_type for s in self.srcs])

    def get_brightest(self, object_type='star', num_srcs=1, band='r', return_idx=False):
        """return brightest sources (by source type, band)"""
        fluxes      = np.array([s.params.flux_dict[band] for s in self.srcs])
        type_idx    = np.where(self.source_types == object_type)[0]
        type_fluxes = fluxes[type_idx]
        type_idx    = type_idx[np.argsort(type_fluxes)[::-1]][:num_srcs]
        blist       = [self.srcs[i] for i in type_idx]
        if return_idx:
            return blist, type_idx
        else:
            return blist

    ####################
    # Resample methods #
    ####################

    def resample_model(self):
        """ resample each field """
        for field in pyprind.prog_bar(self.field_list):
            field.resample_photons(self.srcs)
        self.resample_sources()

    def resample_sources(self):
        for src in pyprind.prog_bar(self.srcs):
            src.resample()

    #####################
    # Plotting Methods  #
    #####################
    def render_model_image(self, roi=None):
        raise NotImplementedError


class Field(object):
    """ holds image data associated with a single field """
    def __init__(self, img_dict):
        self.img_dict = img_dict

        # set each image noise level to the median
        for k, img in self.img_dict.iteritems():
            img.epsilon = np.median(img.nelec)

        # set the (gamma) prior over noise level
        self.a_0 = 5      # convolution parameter - higher tends to avoid 0
        self.b_0 = .005   # inverse scale parameter

    def resample_photons(self, srcs, verbose=False):
        """resample photons - store source-specific images"""

        # first, clear out old sample images
        for src in srcs:
            src.clear_sample_images()

        # generate per-source sample image patch for each fits image in
        # this field.  keep track of photons due to noise
        noise_sums = {}
        for band, img in self.img_dict.iteritems():
            if verbose:
                print " ... resampling band %s " % band
            samp_imgs, noise_sum = \
                cel_mcmc.sample_source_photons_single_image_cython(
                    img, [s.params for s in srcs]
                )

            # tell each source to keep track of it's source-specific sampled
            # images (and the image it was stripped out of)
            for src, samp_img in zip(srcs, samp_imgs):
                if samp_img is not None:

                    # cache pixel grid for each sample image
                    y_grid = np.arange(samp_img.y0, samp_img.y1, dtype=np.float)
                    x_grid = np.arange(samp_img.x0, samp_img.x1, dtype=np.float)
                    xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
                    pixel_grid = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))
                    src.sample_image_list.append((samp_img, img, pixel_grid))

            # keep track of noise sums
            noise_sums[band] = noise_sum

        # resample noise parameter in each fits image
        for band, img in self.img_dict.iteritems():
            a_n         = self.a_0 + noise_sums[band]
            b_n         = self.b_0 + img.nelec.size
            eps_tmp     = img.epsilon
            img.epsilon = np.random.gamma(a_n, 1./b_n)


class Source(object):
    """ source object - holds current state of markov chain and implements
    several sampling methods"""

    def __init__(self, params, model):
        self.params = params
        self.sample_image_list = []

        # create a bounding box of ~ 20x20 pixels to bound location
        # we're assuming we're within 20 pixels on the first guess...
        self.u_lower = self.params.u - .0025
        self.u_upper = self.params.u + .0025
        self.du      = self.u_upper - self.u_lower

        # kept samples for each source
        self.loc_samps      = []
        self.flux_samps     = []
        self.shape_samps    = []
        self.ll_samps       = []

    def clear_sample_images(self):
        self.sample_image_list = []
        #TODO maybe force a garbage collect?

    @property
    def object_type(self):
        if self.is_star():
            return "star"
        elif self.is_galaxy():
            return "galaxy"
        else:
            return "none"

    def is_star(self):
        return self.params.a == 0

    def is_galaxy(self):
        return self.params.a == 1

    ###############################################################
    # source store their own location, fluxes, and shape samples  #
    ###############################################################
    @property
    def location_samples(self):
        return np.array(self.loc_samps)

    @property
    def flux_samples(self):
        return np.array(self.flux_samps)

    @property
    def shape_samples(self):
        return np.array(self.shape_samps)

    @property
    def loglike_samples(self):
        return np.array(self.ll_samps)

    def store_sample(self):
        self.loc_samps.append(self.params.u.copy())
        self.flux_samps.append(self.params.fluxes.copy())
        self.shape_samps.append(self.params.shape.copy())

    def store_loglike(self):
        self.ll_samps.append(self.log_likelihood())

    def flux_in_image(self, fits_image, fluxes=None):
        """convert flux in nanomaggies to flux in pixel counts for a 
        particular image """
        if fluxes is not None:
            band_i = ['u', 'g', 'r', 'i', 'z'].index(fits_image.band)
            f      = fluxes[band_i]
        else:
            f = self.params.flux_dict[fits_image.band]
        band_flux = (f / fits_image.calib) * fits_image.kappa
        return band_flux

    ##########################################################################
    # likelihood functions (used by resampling methods, can be overridden)   #
    ##########################################################################
    def log_likelihood(self, u=None, fluxes=None, shape=None):
        """ conditional likelihood of source conditioned on photon
        sampled images """
        # args passed in or from source's own params
        u      = self.params.u      if u is None else u
        fluxes = self.params.fluxes if fluxes is None else fluxes
        shape  = self.params.shape  if shape is None else shape

        # for each source image, compute per pixel poisson likelihood term
        ll = 0
        for n, (samp_img, fits_img, pixel_grid) in enumerate(self.sample_image_list):
            # grab the patch the sample_image corresponds to (every other 
            # pixel is a zero value)
            ylim, xlim = (samp_img.y0, samp_img.y1), \
                         (samp_img.x0, samp_img.x1)

            # photon scatter image (from psf and galaxy extent) aligned w/ patch
            psf_ns, _, _ = \
                self.compute_scatter_on_pixels(fits_image=fits_img,
                                               u=u, shape=shape,
                                               xlim=xlim, ylim=ylim,
                                               pixel_grid=pixel_grid)

            # convert parameter flux to fits_image specific photon count flux
            band_flux = self.flux_in_image(fits_img, fluxes=fluxes)
            if psf_ns is None:
                ll_img = - band_flux * np.sum(fits_img.weights)
                ll    += ll_img
                continue

            # compute model patch means and the sum of means outside patch (should be small...)
            model_patch   = band_flux * psf_ns

            # compute poisson likelihood of each pixel - note that 
            # the last term would be sum(model_patch) - model_outside, which is 
            # just equal to band_flux * sum(fits_img.weights)
            mask  = (model_patch > 0.)
            ll_img = np.sum( np.log(model_patch[mask]) * 
                             np.array(samp_img.data)[mask] ) - \
                             band_flux*np.sum(fits_img.weights)

            ### debug
            if np.isnan(ll_img):
                print "NAN"
                print "model patch zeros: ", np.sum(model_patch==0)
                print "band flux ", band_flux

            ll += ll_img
        return ll

    def location_likelihood(self, u):
        return self.log_likelihood(u = u)

    def log_likelihood_isolated(self, u=None, fluxes=None, shape=None):
        """ log likelihood of this source if this is the only patch in the 
        local area.  This is the same as saying the pixel for lambda_m is only
        determined by this source's model and the image noise 
        """
        # args passed in or from source's own params
        u      = self.params.u      if u is None else u
        fluxes = self.params.fluxes if fluxes is None else fluxes
        shape  = self.params.shape  if shape is None else shape

        # for each source image, compute per pixel poisson likelihood term
        ll = 0
        for n, (samp_img, fits_img, pixel_grid) in enumerate(self.sample_image_list):
            # grab the patch the sample_image corresponds to
            ylim, xlim = (samp_img.y0, samp_img.y1), \
                         (samp_img.x0, samp_img.x1)
            data_patch = fits_img.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]

            # photon scatter image (from psf and galaxy extent) aligned w/ patch
            psf_ns, _, _ = \
                self.compute_scatter_on_pixels(fits_image=fits_img,
                                               u=u, shape=shape,
                                               pixel_grid=pixel_grid,
                                               ylim=ylim, xlim=xlim)

            # convert parameter flux to fits_image specific photon count flux
            band_flux = self.flux_in_image(fits_img, fluxes=fluxes)
            assert psf_ns is not None, 'source not overlapping - this is a bug fix me.'
            #if psf_ns is None:
            #    ll_img = - band_flux * np.sum(fits_img.weights)
            #    ll    += ll_img
            #    continue

            # compute model patch means and the sum of means outside patch (should be small...)
            model_patch = band_flux * psf_ns + fits_img.epsilon

            # compute poisson likelihood of each pixel - note that 
            # the last term would be sum(model_patch) - model_outside, which is 
            # just equal to band_flux * sum(fits_img.weights)
            mask  = (model_patch > 0.)
            ll_img = np.sum(np.log(model_patch) * np.array(data_patch)) - np.sum(model_patch)

            ### debug
            if np.isnan(ll_img):
                print "NAN"
                print "model patch zeros: ", np.sum(model_patch==0)
                print "band flux ", band_flux

            ll += ll_img
        return ll

    ##########################
    # resampling methods     #
    ##########################
    def resample(self):
        assert len(self.sample_image_list) != 0, "resample source needs sampled images"
        self.resample_fluxes()
        self.resample_location()
        #self.resample_shape()
        #self.resample_type()

    def resample_type(self):
        #TODO: call reversible jump move
        """ resample type of source - star vs. galaxy """
        pass

    def resample_location(self, u=None):
        """ conditionally resample location of source """
        if u is None:
            u = self.params.u.copy()
        u, ll = slicesample(init_x   = u,
                            logprob  = lambda u: self.location_likelihood(u),
                            step     = self.du/5,
                            step_out = False,
                            upper_bound = self.u_upper,
                            lower_bound = self.u_lower)
        self.params.u = u
        return u

    def resample_shape(self):
        """ resamples shape/extent of source (only applicable to galaxies) """
        if self.is_star():
            return
        #TODO implement shape resampler

    def resample_fluxes(self):
        """ samples fluxes, u,g,r,i,z, given all other parameters
              p(b | z, theta) \propto p(z | b, theta) p(b | theta)
                              =       pois(sum z | b, theta) p(b | theta)
        """
        #TODO make the prior over fluxes modular
        # prior params
        a_0 = 5.     # convolution parameter - higher tends to avoid 0
        b_0 = .005   # inverse scale parameter

        # sum detectable photons in sample images
        bands       = ['u', 'g', 'r', 'i', 'z']
        band_counts = {b: 0 for b in bands}
        psf_sums    = {b: 0 for b in bands}
        for src_img, fits_img, pixel_grid in self.sample_image_list:
            band_counts[fits_img.band] += np.sum(np.array(src_img.data))
            psf_ns, ylim, xlim = self.compute_scatter_on_pixels(fits_img, pixel_grid)
            psf_sums[fits_img.band] += np.sum(psf_ns) * fits_img.kappa/fits_img.calib

        # resample fluxes
        a_n    = a_0 + np.array([band_counts[b] for b in bands])
        b_n    = b_0 + np.array([psf_sums[b] for b in bands])
        self.params.fluxes = np.random.gamma(a_n, 1./b_n)

    def compute_scatter_on_pixels(self, fits_image, u=None, shape=None, 
                                        xlim=None, ylim=None,
                                        pixel_grid=None):
        """ compute how photons will be scattered spatially on fits_image, 
        subselecting only the pixels with  > epsilon probability of seeing a 
        photon.
        For a star, this is just the PSF image.  For a Galaxy, this
        is the convolution of the PSF model with the PSF

        kwargs: 
          u          : source (ra, dec) location
          shape      : galaxy shape parameters
          xlim       : pixel limits to compute scatter (must be within fits_image bounds)
          ylim       : ''
          pixel_grid : list of points to evaluate mog (cached for speed)
        """
        #TODO make sure this returns an image with EPSILON error - 
        if u is None:
            u = self.params.u
        if self.is_star():
            patch, ylim, xlim = \
                celeste.gen_point_source_psf_image(u, fits_image,
                                                   xlim=xlim, ylim=ylim,
                                                   pixel_grid=pixel_grid)
            return patch, ylim, xlim
        elif self.is_galaxy():
            if shape is None:
                shape = self.params.shape
            patch, ylim, xlim = \
                gal_funs.gen_galaxy_psf_image(shape, u, fits_image,
                                              xlim=xlim, ylim=ylim,
                                              check_overlap=True,
                                              unconstrained=False,
                                              return_patch=True)
            return patch, ylim, xlim
        else:
            raise NotImplementedError, "only stars and galaxies have photon scattering images"

    def compute_model_patch(self, fits_image, u=None, xlim=None, ylim=None):
        patch, ylim, xlim = \
            self.compute_scatter_on_pixels(fits_image, u=u, xlim=xlim, ylim=ylim)
        band_flux = (self.params.flux_dict[fits_image.band] / fits_image.calib) * \
                    fits_image.kappa
        return band_flux * patch, ylim, xlim

    ###################
    # source plotting #
    ###################
    def plot(self, fits_image, ax, data_ax=None, diff_ax=None):
        import matplotlib.pyplot as plt; import seaborn as sns;
        from CelestePy.util.misc import plot_util
        patch, ylim, xlim = self.compute_model_patch(fits_image)
        cim = ax.imshow(patch, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
        plot_util.add_colorbar_to_axis(ax, cim)
        ax.set_title("model")

        if data_ax is not None:
            dpatch = fits_image.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]].copy()
            print "Data patch median: ", np.median(dpatch)
            dpatch -= np.median(dpatch)
            dpatch[dpatch<0] = 0.
            dim = data_ax.imshow(dpatch, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
            plot_util.add_colorbar_to_axis(data_ax, dim)
            data_ax.set_title("data")

        if diff_ax is not None:
            dpatch = fits_image.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]].copy()
            dpatch -= np.median(dpatch)
            dpatch[dpatch<0] = 0.
            dim = diff_ax.imshow((dpatch - patch), extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
            plot_util.add_colorbar_to_axis(diff_ax, dim)
            msqe  = np.mean((dpatch - patch)**2)
            smsqe = np.mean((dpatch - patch)**2 / patch)
            diff_ax.set_title("diff, mse = %2.3f; chi-sq = %2.3f"%(msqe, smsqe))


#####################################
# Instantiate Generic Celeste class #
#####################################
class Celeste(CelesteBase):
    _source_type = Source

