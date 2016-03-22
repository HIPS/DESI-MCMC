import autograd.numpy as np
import autograd.numpy.linalg as npla
import autograd.scipy.misc as scpm

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


def mog_samples(N, means, chols, pis):
    K, D = means.shape
    indices = discrete(pis, (N,))
    n_means = means[indices,:]
    n_chols = chols[indices,:,:]
    white   = npr.randn(N,D)
    color   = np.einsum('ikj,ij->ik', n_chols, white)
    return color + n_means


class MixtureOfGaussians(object):
    """Evaluate logpdf and sample from Mixture of Gaussians"""

    def __init__(self, means, covs, pis):
        # dimension check
        self.K, self.D = means.shape

        # cache means, covs, pis
        self.update_params(means, covs, pis)

    def update_params(self, means, covs, pis):
        assert covs.shape[1] == covs.shape[2] == self.D
        assert self.K == covs.shape[0] == len(pis), "%d != %d != %d"%(self.K, covs.shape[0], len(pis))
        assert np.isclose(np.sum(pis), 1.)
        self.means = means
        self.covs  = covs
        self.pis   = pis
        self.dets  = np.array([npla.det(c) for c in self.covs])
        self.icovs = np.array([npla.inv(c) for c in self.covs])
        self.chols = np.array([npla.cholesky(c) for c in self.covs])

    def logpdf(self, x):
        return mog_loglike(x, means=self.means, icovs=self.icovs,
                           dets=self.dets,   pis  = self.pis)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self, x):
        return np.dot(self.pis, self.means)

    def var(self, x):
        return np.sum(self.covs * pis[:,None,None], axis=0)

    def rvs(self, size=1):
        return mog_samples(N, self.means, self.chols, self.pis)


