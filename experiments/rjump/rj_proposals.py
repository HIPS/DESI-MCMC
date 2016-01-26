import matplotlib.pyplot     as plt
import autograd.numpy.linalg as npla
import autograd.numpy        as np
import autograd.numpy.random as npr
import autograd.scipy.misc   as scpm
from autograd import grad

import mcmc
from scipy.stats.distributions import gamma


############################################################################
# Likelihoods of varying shapes/dimensionality for testing samplers
############################################################################
def mog_loglike(x, means, icovs, dets, pis):
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
        return logprob[0]
    else:
        return logprob


if __name__=="__main__":

    # galaxy parameters: loc, shape, etc in arg here
    # TODO: actually make this reflect how 
    #       galaxy location, shape parameters actually create gaussian 
    #       mixture parameters
    def gen_mog_params(loc, shape, etc):
        means = np.random.randn(3, 2) + loc
        pis   = np.array([.2, .3, .5])
        covs  = shape* np.array([ [[1, 0], [0, 1]],
                             [[1, .2], [.2, 1]],
                             [[1, .5], [.5, 1]] ])
        icovs = np.array([npla.inv(c) for c in covs])
        dets  = np.array([npla.det(c) for c in covs])
        chols = np.array([npla.cholesky(c) for c in covs])
        return means, covs, icovs, dets, chols, pis

    def gen_model_image(pixel_grid, loc, shape):
        # generate MoG params
        means, covs, icovs, dets, chols, pis = \
            gen_mog_params(loc = loc, shape=shape, etc=None)

        # compute mog likelihod
        loglikes = mog_loglike(pixel_grid, means, icovs, dets, pis)
        return loglikes

    # pixel grid
    xgrid, ygrid = np.arange(50), np.arange(50)
    yy, xx       = np.meshgrid(ygrid, xgrid)
    pixel_grid   = np.column_stack([yy.ravel(), xx.ravel()])

    # visualize
    M1 = gen_model_image(pixel_grid, loc=26., shape=4.)
    M2 = gen_model_image(pixel_grid, loc=24., shape=6.)
    fig, axarr = plt.subplots(1, 2)
    axarr[0].contourf(xx, yy, np.exp(M1).reshape(xx.shape))
    axarr[1].contourf(xx, yy, np.exp(M2).reshape(xx.shape))
    plt.show()

    # define differentiable loss function between two model images
    def loss(th0, th1):
        """ define squared error loss function """
        M1 = gen_model_image(pixel_grid, loc=th0[0], shape=th0[1])
        M2 = gen_model_image(pixel_grid, loc=th1[0], shape=th1[1])
        return np.sum(M1 - M2)

    # compute loss, compute derivative wrt first param
    print loss([23, 2], [26, 4])
    print grad(loss, argnum=0)(np.array([25., 2.]), np.array([26., 4.]))


