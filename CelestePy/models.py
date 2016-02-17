import autograd.numpy as np
import CelestePy.celeste_mcmc as cel_mcmc

class Celeste:
    """ celeste model - base class """
    def __init__(self, gal_flux_prior_distn=None, star_flux_prior_distn=None):
        # model keeps track of a field list
        self.field_list = []

    def initialize_sources(self, init_srcs=None):
        """ initialize sources after adding fields """
        if init_srcs is None:
            init_srcs = self.initialize_from_fields()
        self.srcs = init_srcs

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


class Field:
    """ data and model associated with a single field """
    def __init__(self, img_dict):
        self.img_dict = img_dict

        # set each image noise level to the median
        for k, img in self.img_dict.iteritems():
            img.epsilon = np.median(img.nelec)

    def resample_photons(self, srcs):
        """resample photons - store source-specific images"""
        self.source_imgs = {}
        for band, img in self.img_dict.iteritems():
            samp_imgs, noise_sum = \
                cel_mcmc.sample_source_photons_single_image_cython(img, srcs)
            self.source_imgs[band] = samp_imgs

    def model_images(self):
        pass



