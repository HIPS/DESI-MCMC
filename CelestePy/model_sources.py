"""
Celeste Source Derived Classes

"""
from models import Source, CelesteBase
import autograd.numpy as np
from autograd import grad
import cPickle as pickle

#############################
#source parameter priors    #
#############################
import os
prior_param_dir = os.path.join(os.path.dirname(__file__),
                               '../experiments/empirical_priors')
star_flux_mog = pickle.load(open(os.path.join(prior_param_dir, 'star_fluxes_mog.pkl'), 'rb'))
gal_flux_mog  = pickle.load(open(os.path.join(prior_param_dir, 'gal_fluxes_mog.pkl'), 'rb'))
gal_shape_mog = pickle.load(open(os.path.join(prior_param_dir, 'gal_shape_mog.pkl'), 'rb'))


class SourceGMMPrior(Source):
    def __init__(self, params, model):
        super(SourceGMMPrior, self).__init__(params, model)

    def resample(self):
        assert len(self.sample_image_list) != 0, "resample source needs sampled source images"
        if self.is_star():
            self.resample_star()
        elif self.is_galaxy():
            self.resample_galaxy()


    def resample_star(self):

        # jointly resample fluxes and location
        th = np.concatenate([self.params.u, self.params.fluxes])
        loglike  = lambda th: self.log_likelihood(u=th[:2], fluxes=th[2:])

        def gloglike(th):
            g = np.zeros(th.shape)
            for i in xrange(len(th)):
                de = np.zeros(th.shape); de[i] = 1e-5
                g[i] = (loglike(th + de) - loglike(th - de)) / (2*de[i])
            return g

        print loglike(th)
        print gloglike(th)

        #print "initial conditional likelihood: %2.4f"%loglike(th)
        from scipy.optimize import minimize
        res = minimize(lambda th: -1.*loglike(th), x0=th, method='Nelder-Mead')
        #print "final conditional likelihood: %2.4f"%loglike(res.x)
        self.params.u = res.x[:2]
        self.params.fluxes = res.x[2:]


    def resample_galaxy(self):
        raise NotImplementedError


# Create universe model with this source type
class CelesteGMMPrior(CelesteBase):
    _source_type = SourceGMMPrior

