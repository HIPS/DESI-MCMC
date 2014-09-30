import numpy as np
from scipy.misc import logsumexp

def multivariate_normal_logpdf(x, mean, cov):
    k = np.shape(mean)[0]
    inv_cov = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    x_shift = x - mean
    quad_form = np.dot(np.dot(x_shift, inv_cov), np.transpose(x_shift))
    return -0.5 * (np.log((2*np.pi)**k * det) + quad_form)

def multivariate_normal_pdf(x, mean, cov):
    k = np.shape(mean)[0]
    inv_cov = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    x_shift = x - mean
    quad_form = np.dot(np.dot(x_shift, inv_cov), np.transpose(x_shift))
    return 1 / (np.sqrt((2*np.pi)**k * det)) * np.exp(-0.5 * quad_form)

def gmm_log_like(x, ws, mus, sigs):
    """ Gaussian Mixture Model likelihood
        Input:
          - x    = N x D array of data (N iid)
          - ws   = K length vector that sums to 1, mixing weights
          - mus  = K x D array of mixture component means
          - sigs = K x D x D array of mixture component covariances

        Output:
          - N length array of log likelihood values

        TODO: speed this up
    """
    N_elem = np.atleast_1d(x).shape[0] # number of rows of data
    ll = np.zeros((N_elem, len(ws)))
    for k in range(len(ws)):
        for n in range(N_elem):
            ll[n, k] = multivariate_normal_logpdf(x[n,:], mean=mus[k,:], cov=sigs[k,:,:]) + np.log(ws[k])
        #ll[:, k] = multivariate_normal(mean=mus[k,:], cov=sigs[k,:,:]).logpdf(x) + np.log(ws[k])
    return logsumexp(ll, axis=1)


if __name__ == "__main__":
  # Vis test to make sure GMM is sensible
  import matplotlib.pyplot as plt
  np.random.seed(41)
  K = 3
  means = 2*np.random.randn(K*2).reshape(K, 2)
  covs  = np.zeros((K, 2, 2))
  for i in range(K): 
      covs[i,:,:] = np.random.randn(2,2)
      covs[i,:,:] = covs[i,:,:].dot(covs[i,:,:].T)
  ws  = np.random.rand(K)
  ws /= np.sum(ws)
  print means, covs, ws

  xgrid  = np.linspace(-3, 3, 100)
  ygrid  = np.linspace(-5, 5, 100)
  xx, yy = np.meshgrid(xgrid, ygrid)
  X      = np.column_stack((xx.ravel(), yy.ravel()))
  zz     = gmm_log_like(X, ws,  means, covs).reshape(xx.shape)
  fig, axarr = plt.subplots(1, 2)
  axarr[0].contour(xx, yy, np.exp(zz))

  ## unstable test for comparison
  ll = np.zeros((X.shape[0], len(ws)))
  for k in range(len(ws)):
    for n in range(X.shape[0]):
      ll[n,k] = multivariate_normal_pdf(X[n, :], mean=means[k,:], cov=covs[k,:,:])*ws[k]
  ll = ll.sum(axis=1).reshape(xx.shape)
  axarr[1].contour(xx, yy, ll)

  plt.show()
