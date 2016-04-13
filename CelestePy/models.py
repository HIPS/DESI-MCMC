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

def poisson_loglike(data, model_img):
    mask   = (model_img > 0.)
    ll_img = np.sum(np.log(model_img[mask]) * data[mask]) - \
             np.sum(model_img)
    return ll_img

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
    def render_model_image(self, fimg, xlim=None, ylim=None, exclude=None):
        # create model image, and add each patch in - init with sky noise
        mod_img     = np.ones(fimg.nelec.shape) * fimg.epsilon
        source_list = [s for s in self.srcs if s is not exclude]

        # add each source's model patch
        for s in pyprind.prog_bar(source_list):
            patch, ylim, xlim = s.compute_model_patch(fits_image=fimg, xlim=xlim, ylim=ylim)
            mod_img[ylim[0]:ylim[1], xlim[0]:xlim[1]] += patch

        if xlim is not None and ylim is not None:
            mod_img = mod_img[ylim[0]:ylim[1], xlim[0]:xlim[1]]

        return mod_img

    def img_log_likelihood(self, fimg, mod_img=None):
        if mod_img is None:
            mod_img = self.render_model_image(fimg)
        ll = np.sum(np.log(mod_img) * fimg.nelec) - np.sum(mod_img)
        return ll

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


#####################################
# Instantiate Generic Celeste class #
#####################################
from sources import Source
class Celeste(CelesteBase):
    _source_type = Source

