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

def contains(pt, lower, upper):
    return np.all( (pt > lower) & (pt < upper) )

class SourceGMMPrior(Source):
    def __init__(self, params, model):
        super(SourceGMMPrior, self).__init__(params, model)

    def location_logprior(self, u):
        if contains(u, self.u_lower, self.u_upper):
            return 0.
        else:
            return -np.inf

    def resample(self):
        assert len(self.sample_image_list) != 0, "resample source needs sampled source images"
        if self.is_star():
            self.resample_star()
        elif self.is_galaxy():
            self.resample_galaxy()

    def constrain_loc(self, u_unc):
        u_unit = 1./(1. + np.exp(-u_unc))
        return u_unit * (self.u_upper - self.u_lower) + self.u_lower

    def unconstrain_loc(self, u):
        assert contains(u, self.u_lower, self.u_upper), "point not contained in initial interval!"
        # convert to unit interval, and then apply logit transformation
        u_unit = (u - self.u_lower) / (self.u_upper - self.u_lower)
        return np.log(u_unit) - np.log(1. - u_unit)

    def constrain_shape(self, lg_shape):
        lg_theta, lg_sigma, lg_phi, lg_rho = lg_shape
        theta = 1./(1. + np.exp(-lg_theta))
        sigma = np.exp(lg_sigma)
        phi   = 1./(1. + np.exp(-lg_phi)) * (180) + -180
        rho   = 1./(1. + np.exp(-lg_rho))
        return np.array([theta, sigma, phi, rho])

    def unconstrain_shape(self, shape):
        theta, sigma, phi, rho = shape
        lg_theta = np.log(theta) - np.log(1. - theta)
        lg_sigma = np.log(sigma)
        phi_unit = (phi+180) / 180.
        lg_phi   = np.log(phi_unit) - np.log(1. - phi_unit)
        lg_rho   = np.log(rho) - np.log(1. - rho)
        return np.array([lg_theta, lg_sigma, lg_phi, lg_rho])

    def resample_star(self):
        # jointly resample fluxes and location
        def loglike(th):
            u, color = self.constrain_loc(th[:2]), th[2:]  #unpack params
            fluxes   = np.exp(star_flux_mog.to_fluxes(color))
            ll       = self.log_likelihood(u=u, fluxes=fluxes)
            ll_color = star_flux_mog.logpdf(color)
            return ll+ll_color
        gloglike = grad(loglike)

        # pack params (make sure we convert to color first
        lfluxes = np.log(self.params.fluxes)
        th  = np.concatenate([self.unconstrain_loc(self.params.u),
                              star_flux_mog.to_colors(lfluxes)])
        print "initial conditional likelihood: %2.4f"%loglike(th)
        from scipy.optimize import minimize
        res = minimize(fun = lambda th: -1.*loglike(th),
                       jac = lambda th: -1.*gloglike(th),
                       x0=th,
                       method='L-BFGS-B',
                       options={'ftol' : 1e3 * np.finfo(float).eps})

        print res
        print "final conditional likelihood: %2.4f"%loglike(res.x)
        print gloglike(res.x)
        self.params.u      = self.constrain_loc(res.x[:2])
        self.params.fluxes = np.exp(star_flux_mog.to_fluxes(res.x[2:]))

    def resample_galaxy(self):
        # gradient w.r.t fluxes
        def loglike(th):
            # unpack location, color and shape parameters
            u, color, shape = self.constrain_loc(th[:2]), th[2:7], \
                              self.constrain_shape(th[7:])
            fluxes          = np.exp(gal_flux_mog.to_fluxes(color))
            ll              = self.log_likelihood(u=u, fluxes=fluxes, shape=shape)
            ll_color        = gal_flux_mog.logpdf(color)
            return ll+ll_color
        gloglike = grad(loglike)

        #print "initial conditional likelihood: %2.4f"%loglike(th)
        self.params.theta = np.clip(self.params.theta, 1e-6, 1-1e-6)
        th  = np.concatenate([self.unconstrain_loc(self.params.u),
                              gal_flux_mog.to_colors(np.log(self.params.fluxes)),
                              self.unconstrain_shape(self.params.shape)])
        print "initiali th: ", th
        from scipy.optimize import minimize
        res = minimize(fun = lambda th: -1.*loglike(th),
                       jac = lambda th: -1.*gloglike(th),
                       x0  = th,
                       method='L-BFGS-B', options={'disp':1, 'maxiter':10})

        # store new values
        self.params.u      = self.constrain_loc(res.x[:2])
        self.params.fluxes = np.exp(gal_flux_mog.to_fluxes(res.x[2:7]))
        self.params.shape  = self.constrain_shape(res.x[7:])


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
            params.shape  = np.concatenate(([0.5], self.galaxy_shape_prior.rvs(size=1)[0]))
            logprob_shape = self.galaxy_shape_prior.logpdf(params.shape)
            return params, logprob

