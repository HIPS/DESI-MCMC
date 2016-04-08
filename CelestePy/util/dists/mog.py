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
    white   = np.random.randn(N,D)
    color   = np.einsum('ikj,ij->ik', n_chols, white)
    return color + n_means

def discrete(p, shape):
    length = np.prod(shape)
    indices = p.shape[0] - np.sum(np.random.rand(length)[:,np.newaxis] < np.cumsum(p), axis=1)
    return indices.reshape(shape)

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
        #assert np.isclose(np.sum(pis), 1.)
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
        return mog_samples(size, self.means, self.chols, self.pis)

    def convolve(self, mog):
        """ convolve this mixture of gaussians with mog. """
        # compute pairwise mean/cov sums, weights product
        means   = np.reshape(self.means[:,None] + mog.means[None,:], (-1, 2))
        weights = np.reshape(self.pis[:,None]*mog.pis[None,:], (-1,))
        covs    = np.reshape(self.covs[:,None] + mog.covs[None,:], (-1, 2, 2))
        return MixtureOfGaussians(means, covs, weights)


    def apply_affine(self, A, b):
        """ return the distribution of an affine transformation of this mog RV
            affine = Ax + b
        """
        return MixtureOfGaussians(
            means = np.dot(self.means, A.T) + b,
            covs  = np.array([ np.dot(np.dot(A, c), A.T) for c in self.covs ]),
            pis   = self.pis
        )

    @staticmethod
    def convex_combine(mogs, mixing_weights):
        return MixtureOfGaussians(
            means = np.row_stack([mog.means for mog in mogs]),
            covs  = np.row_stack([mog.covs for mog in mogs]),
            pis   = np.concatenate([ w*mog.pis for w, mog in zip(mixing_weights, mogs)])
        )

    def evaluate_grid(self, xlim, ylim, pts=None):
        assert (ylim[1] > ylim[0]) and (xlim[1] > xlim[0]), "bad limits."
        if pts is None:
            y_grid = np.arange(ylim[0], ylim[1], dtype=np.float)
            x_grid = np.arange(xlim[0], xlim[1], dtype=np.float)
            xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
            pts    = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))

        # compute loglike at each point
        lls = mog_loglike(pts, self.means, self.icovs, self.dets, self.pis)
        return np.reshape(np.exp(lls), xx.shape) # (ylim[1]-ylim[0], xlim[1]-xlim[0]))


