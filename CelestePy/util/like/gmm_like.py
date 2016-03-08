import autograd.numpy as np
import autograd.scipy.misc as scpm
from scipy.stats import multivariate_normal

def gmm_logprob(x, ws, mus, sigs, invsigs=None, logdets=None):
    """ Gaussian Mixture Model likelihood
        Input:
          - x    = N x D array of data (N iid)
          - ws   = K length vector that sums to 1, mixing weights
          - mus  = K x D array of mixture component means
          - sigs = K x D x D array of mixture component covariances

          - invsigs = K x D x D array of mixture component covariance inverses
          - logdets = K array of mixture component covariance logdets

        Output:
          - N length array of log likelihood values

        TODO: speed this up
    """

    if sigs is None:
        assert invsigs is not None and logdets is not None, \
                "need sigs if you don't include logdets and invsigs"

    # compute invsigs if needed
    if invsigs is None:
        invsigs = np.array([np.linalg.inv(sig) for sig in sigs])
        logdets = np.array([np.linalg.slogdet(sig)[1] for sig in sigs])

    # compute each gauss component separately
    xx = np.atleast_2d(x)
    centered = xx[:,:,np.newaxis] - mus.T[np.newaxis,:,:]
    solved   = np.einsum('ijk,lji->lki', invsigs, centered)
    logprobs = -0.5*np.sum(solved * centered, axis=1) - \
                    np.log(2*np.pi) - 0.5*logdets + np.log(ws)
    logprob  = scpm.logsumexp(logprobs, axis=1)
    if len(x.shape) == 1:
        return logprob[0]
    else:
        return logprob

def gmm_prob(x, ws, mus, sigs, invsigs=None, logdets=None):
    return np.exp(gmm_logprob(x, ws, mus, sigs, invsigs, logdets))

def mog_logmarglike(x, means, covs, pis, ind=0):
    """ marginal x or y (depending on ind) """
    K = pis.shape[0]
    xx = np.atleast_2d(x)
    centered = xx.T - means[:,ind,np.newaxis].T
    logprobs = []
    for kk in xrange(K):
        quadterm  = centered[:,kk] * centered[:,kk] * (1./covs[kk,ind,ind])
        logprobsk = -.5*quadterm - .5*np.log(2*np.pi) \
                    -.5*np.log(covs[kk,ind,ind]) + np.log(pis[kk])
        logprobs.append(np.squeeze(logprobsk))
    logprobs = np.array(logprobs)
    logprob  = scpm.logsumexp(logprobs, axis=0)
    if np.isscalar(x):
        return logprob[0]
    else:
        return logprob 


##################
# MVN Funcs      #
##################
log2pi = np.log(2.*np.pi)
def multivariate_normal_logpdf(x, mean, cov, logdet=None, covinv=None):
    """ Log density of a multivariate normal distribution 
        Input:
          - x  : N x D array of data (N iid)
          - mu : D length vector, mean values
          - cov: D x D array, covariance

          For tightening up loops:
          - logdet : log determinant of the covariance matrix
          - covinv : inverse of the covariance (for speed-ups)
    """
    D = mean.shape[0]

    # compute quadratic term, (x-mu)^T sig^{-1} (x-mu)
    if covinv is None:
        covinv = np.linalg.inv(cov)
    #x_shift    = x - mean
    #quad_form  = (x_shift * covinv.dot(x_shift.T).T).sum(axis=1)

    # quick quad form?
    x_sig     = x.dot(covinv)
    quad_form = (x * x_sig).sum(axis=1) - 2. * x_sig.dot(mean) + mean.dot(covinv).dot(mean)
    if logdet is None:
        sign, logdet = np.linalg.slogdet(cov)
    return -.5 * D * log2pi - .5 * logdet - 0.5 * quad_form


def multivariate_normal_pdf(x, mean, cov, logdet=None, covinv=None):
    return np.exp(multivariate_normal_logpdf(x, mean, cov, logdet, covinv))


