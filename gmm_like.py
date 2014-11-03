import numpy as np
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal

# cache used value
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

def gmm_like(x, ws, mus, sigs, invsigs=None, logdets=None):
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
    N_elem = np.atleast_1d(x).shape[0] # number of rows of data
    ll = np.zeros((N_elem, len(ws)))
    for k in range(len(ws)):
        if invsigs is not None and logdets is not None: 
            ll[:, k] = multivariate_normal_pdf(x,
                                               mean = mus[k,:],
                                               cov  = sigs[k,:,:],
                                               logdet = logdets[k],
                                               covinv = invsigs[k,:,:]) * ws[k] # + np.log(ws[k])
        else:
            ll[:, k] = multivariate_normal_pdf(x,
                                               mean = mus[k,:],
                                               cov  = sigs[k,:,:]) * ws[k] #+ np.log(ws[k])
        #ll[:,k] = multivariate_normal(mean = means[k,:],
        #                              cov  = covs[k,:,:]).logpdf(X) + np.log(ws[k])
    return ll.sum(axis=1)

def gmm_log_like(x, ws, mus, sigs, invsigs=None, logdets=None):
    return np.log(gmm_like(x, ws, mus, sigs, invsigs, logdets))
    #return logsumexp(ll, axis=1)

if __name__ == "__main__":
  # Vis test to make sure GMM is sensible
  import matplotlib.pyplot as plt
  np.random.seed(41)
  K = 3
  means = 2*np.random.randn(K*2).reshape(K, 2)
  covs  = np.zeros((K, 2, 2))
  invcovs = np.zeros((K, 2, 2))
  logdets = np.zeros(K)
  for i in range(K): 
      covs[i,:,:] = np.random.randn(2,2)
      covs[i,:,:] = covs[i,:,:].dot(covs[i,:,:].T)
      invcovs[i,:,:] = np.linalg.inv(covs[i,:,:])
      sign, logdet = np.linalg.slogdet(covs[i,:,:])
      logdets[i] = logdet
  ws  = np.random.rand(K)
  ws /= np.sum(ws)

  #########################################################
  # Generate grid of points for testing
  #########################################################
  xgrid  = np.linspace(-3, 3, 100)
  ygrid  = np.linspace(-5, 5, 100)
  xx, yy = np.meshgrid(xgrid, ygrid)
  X      = np.column_stack((xx.ravel(), yy.ravel()))

  ######################################################
  # Test multivariate normal log pdf
  ######################################################
  #covs[2,:,:] = 2 * np.eye(2)
  ll_slow = multivariate_normal(mean = means[2,:], 
                                cov  = covs[2,:,:]).logpdf(X)
  ll_fast = multivariate_normal_logpdf(X,
                                       mean = means[2,:],
                                       cov  = covs[2,:,:])

  # check inverse cov version
  sign, logdet = np.linalg.slogdet(covs[2,:,:])
  ll_faster = multivariate_normal_logpdf(X,
                                         mean = means[2,:], 
                                         cov  = covs[2,:,:],
                                         covinv = np.linalg.inv(covs[2,:,:]),
                                         logdet = logdet)
  if np.allclose(ll_slow, ll_fast, atol=1e-6):
    print "Fast MVN log pdf passes"
  else: 
    print "Fast MVN log pdf fails", np.mean(np.abs(ll_slow - ll_fast))

  if np.allclose(ll_slow, ll_faster, atol=1e-6):
    print "Faster (with inv cov caching) MVN log pdf passes"
  else:
    print "Faster (with inv cov caching) MVN log pdf fails!!!!", np.mean(np.abs(ll_slow - ll_fast))

  ########################################################
  # Test multivariante MIXTURE of Gaussians
  ########################################################
  ll_fast = gmm_log_like(X, ws,  means, covs)
  ll_faster = gmm_log_like(X, ws, means, covs,
                           invsigs = invcovs,
                           logdets = logdets)

  # slow and steady
  N_elem = np.atleast_1d(X).shape[0]
  ll = np.zeros((N_elem, len(ws)))
  for k in range(K):
      ll[:,k] = multivariate_normal(mean = means[k,:],
                                    cov  = covs[k,:,:]).logpdf(X) + np.log(ws[k])
  ll_slow = logsumexp(ll, axis=1)

  if np.allclose(ll_fast, ll_slow, atol=1e-6): 
    print "GMM slow vs. fast test passes"
  else:
    print "GMM slow vs. fast test FAILS!!!!!"

  if np.allclose(ll_faster, ll_slow, atol=1e-6):
    print "GMM slow vs. faster test passes"
  else:
    print "GMM slow vs. faster test FAILS !!!!!"


  #### profile a little
  #%lprun -m gmm_like \
  #ll_faster = gmm_like(X, ws, means, covs, invsigs = invcovs, logdets = logdets)

  ## unstable test for comparison
  #ll = np.zeros((X.shape[0], len(ws)))
  #for k in range(len(ws)):
  #  for n in range(X.shape[0]):
  #    ll[n,k] = multivariate_normal_logpdf(X[n, :], mean=means[k,:], cov=covs[k,:,:])*ws[k]
  #ll = ll.sum(axis=1).reshape(xx.shape)
  #axarr[1].contour(xx, yy, np.exp(ll))

  #fig, axarr = plt.subplots(1, 2)
  #zz = ll_fast.reshape(xx.shape)
  #axarr[0].contour(xx, yy, np.exp(zz))
  #plt.show()
