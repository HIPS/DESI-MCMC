import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('..'), os.path.pardir)))
from gmm_like import multivariate_normal_logpdf, gmm_log_like
import numpy as np
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import gmm_like_fast as gmlf

def test_multivariate_normal_log_pdf():
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

    ## Generate grid of points for testing
    xgrid  = np.linspace(-3, 3, 100)
    ygrid  = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(xgrid, ygrid)
    X      = np.column_stack((xx.ravel(), yy.ravel()))

    #covs[2,:,:] = 2 * np.eye(2)
    ll_slow = multivariate_normal(mean = means[2,:], 
                                  cov  = covs[2,:,:]).logpdf(X)
    ll_fast = multivariate_normal_logpdf(X,
                                         mean = means[2,:],
                                         cov  = covs[2,:,:])

    ll_faster = np.zeros(X.shape[0]); \
        gmlf.multivariate_normal_2d_covinv_logdet_logpdf(
            lls    = ll_faster,
            x      = X,
            mean   = means[2,:],
            covinv = invcovs[2, :, :],
            logdet = logdets[2])

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


def test_gmm_like():
    np.random.seed(41)
    K = 10
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

    ## Generate grid of points for testing
    xgrid  = np.linspace(-3, 3, 200)
    ygrid  = np.linspace(-5, 5, 200)
    xx, yy = np.meshgrid(xgrid, ygrid)
    X      = np.column_stack((xx.ravel(), yy.ravel()))

    ## test different methods
    ll_fast   = gmm_log_like(X, ws,  means, covs)
    ll_faster = gmm_log_like(X, ws, means, covs,
                             invsigs = invcovs,
                             logdets = logdets)

    ll_fastest = np.zeros(X.shape[0]); \
        gmlf.gmm_like_2d_covinv_logdet(probs = ll_fastest, 
                                       x     = X,
                                       ws    = ws,
                                       mus   = means,
                                       invsigs = invcovs,
                                       logdets = logdets)

    ## slow and steady
    N_elem = np.atleast_1d(X).shape[0]
    ll = np.zeros((N_elem, len(ws)))
    for k in range(K):
        ll[:,k] = multivariate_normal(mean = means[k,:],
                                      cov  = covs[k,:,:]).logpdf(X) + np.log(ws[k])
    ll_slow = logsumexp(ll, axis=1)
    assert np.allclose(ll_fast, ll_slow), "fast doesn't match slow"
    assert np.allclose(ll_faster, ll_fast), "faster doesn't match faster"
    assert np.allclose(ll_faster, np.log(ll_fastest)), "faster doesn't match cython fastest"
    print "GMM LIKE TESTS PASS"

if __name__ == "__main__":
    test_multivariate_normal_log_pdf()
    test_gmm_like()
