"""
CelestePy catalog of the universe.

Andrew Miller <acm@seas.harvard.edu>
"""

import autograd.numpy as np
import CelestePy.celeste_mcmc as cel_mcmc
import CelestePy.util.data as du
from  CelestePy import celeste
import CelestePy.celeste_galaxy_conditionals as gal_funs
import pyprind

class Celeste:
    """ Main model class - interface to user.  Holds a list of
    Source objects, each of which contains local markov chain state, 
    including source parameters, source-sample images"""
    def __init__(self, gal_flux_prior_distn=None, star_flux_prior_distn=None):
        #TODO incorporate priors over fluxes
        # model keeps track of a field list
        self.field_list = []
        self.bands = ['u', 'g', 'r', 'i', 'z']

    def initialize_sources(self, init_srcs=None, init_src_params=None, photoobj_df=None):
        """ initialize sources after adding fields """
        if init_srcs is not None:
            self.srcs = init_srcs
        elif init_src_params is not None:
            self.srcs = [Source(s) for s in init_src_params]
        elif photoobj_df is not None:
            self.srcs = [Source(du.photoobj_to_celestepy_src(p)) for (i, p) in photoobj_df.iterrows()]
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

    def resample_model(self):
        """ resample each field """
        for field in pyprind.prog_bar(self.field_list):
            field.resample_photons(self.srcs)
        self.resample_sources()

    def resample_sources(self):
        for src in pyprind.prog_bar(self.srcs):
            src.resample()

    def render_model_image(self, roi=None):
        raise NotImplementedError


class Field:
    """ holds image data associated with a single field """
    def __init__(self, img_dict):
        self.img_dict = img_dict

        # set each image noise level to the median
        for k, img in self.img_dict.iteritems():
            img.epsilon = np.median(img.nelec)

        # set the (gamma) prior over noise level
        self.a_0 = 5      # convolution parameter - higher tends to avoid 0
        self.b_0 = .005   # inverse scale parameter

    def resample_photons(self, srcs):
        """resample photons - store source-specific images"""

        # first, clear out old sample images
        for src in srcs:
            src.clear_sample_images()

        # generate per-source sample image patch for each fits image in
        # this field.  keep track of photons due to noise
        noise_sums = {}
        for band, img in self.img_dict.iteritems():
            print " ... resampling band %s " % band
            samp_imgs, noise_sum = \
                cel_mcmc.sample_source_photons_single_image_cython(img, [s.params for s in srcs])

            # tell each source to keep track of it's source-specific sampled
            # images (and the image it was stripped out of)
            for src, samp_img in zip(srcs, samp_imgs):
                if samp_img is not None:
                    src.sample_image_list.append((samp_img, img))

            # keep track of noise sums
            noise_sums[band] = noise_sum

        # resample noise parameter in each fits image
        for band, img in self.img_dict.iteritems():
            a_n         = self.a_0 + noise_sums[band]
            b_n         = self.b_0 + img.nelec.size
            eps_tmp     = img.epsilon
            img.epsilon = np.random.gamma(a_n, 1./b_n)


class Source:
    """ source object - holds current state of markov chain and implements
    several sampling methods"""

    def __init__(self, params):
        self.params = params
        self.sample_image_list = []

        # create a bounding box of ~ 20x20 pixels to bound location
        # we're assuming we're within 20 pixels on the first guess...
        self.u_lower = self.params.u - .0025
        self.u_upper = self.params.u + .0025
        self.du      = self.u_upper - self.u_lower

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

    def flux_in_image(self, fits_image):
        """convert flux in nanomaggies to flux in pixel counts for a 
        particular image """
        band_flux = (self.params.flux_dict[fits_image.band] / fits_image.calib) * \
                    fits_image.kappa
        return band_flux

    ##########################################################################
    # likelihood functions (used by resampling methods, can be overridden)   #
    ##########################################################################
    def location_likelihood(self, u):
        ll = 0
        for n, (samp_img, fits_img) in enumerate(self.sample_image_list):
            # grab the patch the sample_image corresponds to (everything else is zero)
            ylim, xlim = (samp_img.y0, samp_img.y1), \
                         (samp_img.x0, samp_img.x1)

            # create the scatter image (from psf and galaxy extent) on the patch
            psf_ns, _, _ = \
                self.compute_scatter_on_pixels(fits_image=fits_img, u=u,
                                               ylim=ylim, xlim=xlim)
            # compute the sum over the lamba outside the range
            #lambda_outside = np.sum(fits_img.weights) - np.sum(psf_ns)

            # convert parameter flux to fits_image specific photon count flux
            band_flux   = self.flux_in_image(fits_img)
            if psf_ns is None:
                ll_img = - band_flux * np.sum(fits_img.weights)
                #print "psf_ns is none!", ll_img
                ll    += ll_img
                continue

            # compute model patch means and the sum of means outside patch (should be small...)
            model_patch   = band_flux * psf_ns
            #model_outside = band_flux * lambda_outside

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

    ##########################
    # resampling methods     #
    ##########################
    def resample(self):
        self.resample_fluxes()
        self.resample_location()
        self.resample_shape()
        self.resample_type()

    def resample_type(self):
        #TODO: call reversible jump move
        """ resample type of source - star vs. galaxy """
        pass

    def resample_location(self, u=None):
        """ conditionally resample location of source """
        if u is None:
            u = self.params.u.copy()
        from CelestePy.util.infer.slicesample import slicesample
        u, ll = slicesample(init_x   = u,
                            logprob  = lambda u: self.location_likelihood(u),
                            step     = self.du/5,
                            step_out = False,
                            upper_bound = self.u_upper,
                            lower_bound = self.u_lower)
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
        for src_img, fits_img in self.sample_image_list:
            band_counts[fits_img.band] += np.sum(np.array(src_img.data))
            psf_ns, ylim, xlim = self.compute_scatter_on_pixels(fits_img)
            psf_sums[fits_img.band] += np.sum(psf_ns) * fits_img.kappa/fits_img.calib

        # resample fluxes
        a_n    = a_0 + np.array([band_counts[b] for b in bands])
        b_n    = b_0 + np.array([psf_sums[b] for b in bands])
        self.params.fluxes = np.random.gamma(a_n, 1./b_n)

    def compute_scatter_on_pixels(self, fits_image, epsilon=.0001, u=None, xlim=None, ylim=None):
        """ compute how photons will be scattered spatially on fits_image, 
        subselecting only the pixels with  > epsilon probability of seeing a 
        photon.
        For a star, this is just the PSF image.  For a Galaxy, this
        is the convolution of the PSF model with the PSF
        """
        #TODO make sure this returns an image with EPSILON error - 
        if u is None:
            u = self.params.u
        if self.is_star():
            patch, ylim, xlim = \
                celeste.gen_point_source_psf_image(u, fits_image,
                                                   xlim=xlim, ylim=ylim)
            return patch, ylim, xlim
        elif self.is_galaxy():
            gal_th = np.array([self.params.theta, self.params.sigma,
                               self.params.phi,   self.params.rho])
            patch, ylim, xlim = \
                gal_funs.gen_galaxy_psf_image(gal_th, u, fits_image,
                                              xlim=xlim, ylim=ylim,
                                              check_overlap=True,
                                              unconstrained=False,
                                              return_patch=True)
                #celeste.gen_galaxy_psf_image(self.params, fits_image)
            return patch, ylim, xlim
        else:
            raise NotImplementedError, "only stars and galaxies have photon scattering images"

    def compute_model_patch(self, fits_image, u=None, xlim=None, ylim=None):
        patch, ylim, xlim = \
            self.compute_scatter_on_pixels(fits_image, u=u, xlim=xlim, ylim=ylim)
        band_flux = (self.params.flux_dict[fits_image.band] / fits_image.calib) * \
                    fits_image.kappa
        return band_flux * patch, ylim, xlim

    def plot(self, fits_image, ax, data_ax=None, diff_ax=None):
        import matplotlib.pyplot as plt; import seaborn as sns;
        from CelestePy.util.misc import plot_util
        patch, ylim, xlim = self.compute_model_patch(fits_image)
        cim = ax.imshow(patch, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
        plot_util.add_colorbar_to_axis(ax, cim)

        if data_ax is not None:
            dpatch = fits_image.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            dpatch -= np.median(dpatch)
            dim = data_ax.imshow(dpatch, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
            plot_util.add_colorbar_to_axis(data_ax, dim)

        if diff_ax is not None:
            dpatch = fits_image.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            dpatch -= np.median(dpatch)
            dim = diff_ax.imshow((dpatch - patch), extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
            plot_util.add_colorbar_to_axis(diff_ax, dim)

