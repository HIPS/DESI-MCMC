"""
Celeste Source Derived Classes

"""
from CelestePy.models import Source, CelesteBase
from CelestePy.celeste_src import SrcParams
import autograd.numpy as np
from autograd import grad
import cPickle as pickle

#############################
#source parameter priors    #
#############################
import os
prior_param_dir = os.path.join(os.path.dirname(__file__),
                               '../experiments/empirical_priors')
prior_param_dir = '../empirical_priors/'
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
        gloglike = grad(loglike)
        print loglike(th)
        print gloglike(th)

        #print "initial conditional likelihood: %2.4f"%loglike(th)
        from scipy.optimize import minimize
        res = minimize(fun = lambda th: -1.*loglike(th),
                       x0=th, method='Nelder-Mead')
        #print "final conditional likelihood: %2.4f"%loglike(res.x)
        self.params.u = res.x[:2]
        self.params.fluxes = res.x[2:]


    def resample_galaxy(self):

        # gradient w.r.t fluxes
        colors   = gal_flux_mog.to_colors(self.params.fluxes)
        def color_logprob(c):
            return self.log_likelihood(fluxes=gal_flux_mog.to_fluxes(c)) + \
                   gal_flux_mog.logpdf(c)
        dcolor_logprob = grad(color_logprob)

        print color_logprob(colors)
        print dcolor_logprob(colors)

        #print "initial conditional likelihood: %2.4f"%loglike(th)
        from scipy.optimize import minimize
        res = minimize(fun = lambda th: -1.*color_logprob(th), jac=dcolor_logprob,
                       x0  = colors, method='L-BFGS-B', options={'disp':1})
        self.params.fluxes = gal_flux_mog.to_fluxes(res.x)
        #print "final conditional likelihood: %2.4f"%loglike(res.x)
        #self.params.u = res.x[:2]
        #self.params.fluxes = res.x[2:]



# Create universe model with this source type
class CelesteGMMPrior(CelesteBase):
    _source_type = SourceGMMPrior

    def __init__(self, star_flux_prior   = star_flux_mog,
                       galaxy_flux_prior = gal_flux_mog,
                       galaxy_shape_prior = gal_shape_mog):
        self.star_flux_prior    = star_flux_prior
        self.galaxy_flux_prior  = galaxy_flux_prior
        self.galaxy_shape_prior = galaxy_shape_prior
        super(CelesteGMMPrior, self).__init__()

    def logprior(self, params):
        if params.is_star():
            color = self.star_flux_prior.to_colors(params.fluxes)
            return self.star_flux_prior.logpdf(color)
        elif params.is_galaxy():
            color = self.galaxy_flux_prior.to_colors(params.fluxes)
            return self.galaxy_flux_prior.logpdf(color) + 0.
                    # todo include constraints for shape parameters

    def prior_sample(self, src_type, u=None):
        params = SrcParams(u=u)
        if src_type == 'star':
            # TODO SET a with atoken
            params.a = 0
            color   = self.star_flux_prior.rvs(size=1)[0]
            logprob = self.star_flux_prior.logpdf(color)
            params.fluxes = np.exp(self.star_flux_prior.to_fluxes(color))
            return params, logprob

        elif src_type == 'galaxy':
            params.a = 1
            color   = self.galaxy_flux_prior.rvs(size=1)[0]
            logprob = self.galaxy_flux_prior.logpdf(color)
            params.fluxes = np.exp(self.galaxy_flux_prior.to_fluxes(color))
            params.shape  = self.galaxy_shape_prior.rvs(size=1)[0]
            logprob_shape = self.galaxy_shape_prior.logpdf(params.shape)
            return params, logprob

