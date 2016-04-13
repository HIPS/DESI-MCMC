import autograd.numpy as np
import CelestePy.celeste_galaxy_conditionals as gal_funs
import CelestePy.celeste as celeste

class Source(object):
    """ source object - holds current state of markov chain and implements
    several sampling methods"""

    def __init__(self, params, model):
        self.params = params
        self.model  = model
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

    def __str__(self):
        label = "%s at Ra, Dec = (%2.5f, %2.5f) " % \
            (self.object_type, self.params.u[0], self.params.u[1])
        mags  = " ".join([ "%s = %2.4f"%(b, self.nanomaggies2mags(f))
                           for b,f in zip(['u', 'g', 'r', 'i', 'z'],
                                           self.params.fluxes) ])
        if self.is_star():
            return label + " with Mags " + mags
        else:
            shape = "re=%2.3f, ab=%2.3f, phi=%2.3f" % \
                (self.params.sigma, self.params.rho, self.params.phi)
            return label + " with Mags " + mags + " and Galaxy Shape: " + shape

    def mags2nanomaggies(self, mags):
        return np.power(10., (mags - 22.5)/-2.5)

    def nanomaggies2mags(self, nanos):
        return (-2.5)*np.log10(nanos) + 22.5

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

    @property
    def id(self):
        return "in_%s_to_%s"%(np.str(self.u_lower), np.str(self.u_upper))

    ###############################################################
    # source store their own location, fluxes, and shape samples  #
    ###############################################################
    @staticmethod
    def get_bounding_box(params, img):
        if params.is_star():
            bound = img.R
        elif params.is_galaxy():
            bound = gal_funs.gen_galaxy_psf_image_bound(params, img)
        else:
            raise "source type unknown"
        px, py = img.equa2pixel(params.u)
        xlim = (np.max([0,                  np.floor(px - bound)]),
                np.min([img.nelec.shape[1], np.ceil(px + bound)]))
        ylim = (np.max([0,                  np.floor(py - bound)]),
                np.min([img.nelec.shape[0], np.ceil(py + bound)]))
        return xlim, ylim

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
        assert np.all(~np.isnan(fluxes)), 'passing in NAN fluxes.'

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

    def resample_type(self):
        #TODO: call reversible jump move
        """ resample type of source - star vs. galaxy """
        # propose new parameter setting: 
        #  - return new params, probability of proposal, probability of reverse proposal
        #  - and log det of transformation from current to proposed parameters
        proposal, logpdf, logreverse, logdet = self.propose_other_type()

        fimgs = [self.model.field_list[0].img_dict[b] for b in self.model.bands]
        accept_logprob = self.calculate_acceptance_logprob(
                proposal, logpdf, logreverse, logdet, fimgs)
        if np.log(np.random.rand()) < accept_logprob:
            self.params = proposal
            #TODO: keep stats of acceptance + num like evals

    def propose_other_type(self):
        """ prior-based proposal (simplest).  override this for more
        efficient proposals
        """
        if self.is_star():
            params, logprob = self.model.prior_sample('galaxy', u=self.params.u)
        elif self.is_galaxy():
            params, logprob = self.model.prior_sample('star', u=self.params.u)
        logreverse = self.model.logprior(self.params)
        return params, logprob, logreverse, 0.

    def calculate_acceptance_logprob(self, proposal, logprob_proposal, 
            logprob_reverse, logdet, images):

        def image_like(src, img):
            # get biggest bounding box needed to consider for this image
            xlimp, ylimp = Source.get_bounding_box(params=proposal, img=img)
            xlimc, ylimc = Source.get_bounding_box(params=self.params, img=img)
            xlim = (np.min([xlimp[0], xlimc[0]]), np.max([xlimp[1], xlimc[1]]))
            ylim = (np.min([ylimp[0], ylimc[0]]), np.max([ylimp[1], ylimc[1]]))

            # model image all other sources
            background_img = \
                self.model.render_model_image(img, xlim=xlim, ylim=ylim, exclude=self)
            data_img      = img.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]

            # model image for img, (xlim, ylim)
            model_img, _, _ = src.compute_model_patch(img, xlim=xlim, ylim=ylim)

            # compute current model loglike and proposed model loglike
            ll = poisson_loglike(data      = data_img,
                                 model_img = background_img+model_img)
            return ll

        # compute current and proposal model likelihoods
        curr_like = np.sum([image_like(self, img) for img in images])
        curr_logprior = self.model.logprior(self.params)

        proposal_source = self.model._source_type(proposal, self.model)
        prop_like = np.sum([image_like(proposal_source, img) for img in images])
        prop_logprior = self.model.logprior(proposal_source.params)

        # compute acceptance ratio
        accept_ll = (prop_like + prop_logprior) - (curr_like + curr_logprior) + \
                    (logprob_reverse - logprob_proposal) - \
                    logdet
        return accept_ll

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
    def plot(self, fits_image, ax, data_ax=None, diff_ax=None, unit_flux=False):
        import matplotlib.pyplot as plt; import seaborn as sns;
        from CelestePy.util.misc import plot_util
        if unit_flux:
            patch, ylim, xlim = self.compute_scatter_on_pixels(fits_image)
        else:
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
            diff_ax.set_title("diff, mse = %2.3f"%msqe)


