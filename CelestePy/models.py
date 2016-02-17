"""
CelestePy catalog of the universe.

Andrew Miller <acm@seas.harvard.edu>
"""

import autograd.numpy as np
import CelestePy.celeste_mcmc as cel_mcmc

class Celeste:
    """ Main model class - interface to user.  Holds a list of
    Source objects, each of which contains local markov chain state, 
    including source parameters, source-sample images"""
    def __init__(self, gal_flux_prior_distn=None, star_flux_prior_distn=None):
        #TODO incorporate priors over fluxes
        # model keeps track of a field list
        self.field_list = []

    def initialize_sources(self, init_srcs=None, init_src_params=None):
        """ initialize sources after adding fields """
        if init_srcs is None and init_src_params is None:
            init_srcs = self.initialize_from_fields()
        elif init_srcs is not None:
            self.srcs = init_srcs
        elif init_src_params is not None:
            self.srcs = [Source(s) for s in init_src_params]

    def intialize_from_fields(self):
        """initialize sources from bright spots - make everything a point source?"""
        raise NotImplementedError

    def add_field(self, img_dict):
        """ add a field (run/camcol/field) image information to the model
                img_dict  = fits image keyed by 'ugriz' band
                init_srcs = sources (tractor or celeste) initialized in this field
        """
        self.field_list.append(Field(img_dict))

    def resample_model(self):
        """ resample each field """
        for field in self.field_list:
            field.resample_photons(self.srcs)

        for src in self.srcs:
            src.resample()

    def render_model_image(self):
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

    def clear_sample_images(self):
        self.sample_image_list = []
        #TODO maybe force a garbage collect?

    def is_star(self):
        return self.params.a == 0

    def is_galaxy(self):
        return self.params.a == 1

    def resample(self):
        self.resample_type()
        self.resample_fluxes()
        self.resample_location()
        self.resample_shape()

    def resample_type(self):
        #TODO: call reversible jump move
        """ resample type of source - star vs. galaxy """
        pass

    def resample_location(self):
        """ conditionally resample location of source """
        pass

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
            band_counts[fits_img.band] += np.sum(src_img)
            psf_ns, ylim, xlim = self.compute_scatter_on_pixels(fits_image)
            psf_sums[fits_img.band] += np.sum(psf_ns) # * fits_img.kappa/fits_img.calib

        # resample fluxes
        a_n    = a_0 + np.array([band_counts[b] for b in bands])
        b_n    = b_0 + np.array([psf_sums[b] for b in bands])
        fluxes = np.random.gamma(a_n, 1./b_n)
        self.params.fluxes = dict(zip(bands, fluxes))

    def compute_scatter_on_pixels(self, fits_image, epsilon=.0001):
        """ compute how photons will be scattered spatially on fits_image, 
        subselecting only the pixels with  > epsilon probability of seeing a 
        photon.
        For a star, this is just the PSF image.  For a Galaxy, this
        is the convolution of the PSF model with the PSF
        """
        #TODO make sure this returns an image with EPSILON error - 
        if self.is_star():
            patch, ylim, xlim = \
                celeste.gen_point_source_psf_image(self.params.u, fits_image)
            return patch, ylim, xlim
        elif self.is_galaxy():
            patch, ylim, xlim = \
                celeste.gen_galaxy_psf_image(self.params, fits_image)
            return patch, ylim, xlim
        else:
            raise NotImplementedError, "only stars and galaxies have photon scattering images"
