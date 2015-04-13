import sys
import os.path
from CelestePy.util.like import gmm_like_2d, gmm_like, ein_gmm_like
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import numpy as np

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
    K = 42
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
    ll_fast = gmm_like(X, ws, means, covs)

    #%lprun -m CelestePy.util.like.gmm_like ll_ein = ein_gmm_like(X, ws, means, covs)
    ll_faster = np.zeros(X.shape[0])
    gmm_like_2d(ll_faster, x=X.astype(np.float), 
                           ws=ws.astype(np.float),
                           mus=means.astype(np.float), sigs=covs.astype(np.float))

    # slow and steady
    N_elem = np.atleast_1d(X).shape[0]
    ll = np.zeros((N_elem, len(ws)))
    for k in range(K):
        ll[:,k] = multivariate_normal(mean = means[k,:],
                                      cov  = covs[k,:,:]).logpdf(X) + np.log(ws[k])
    ll_slow = np.exp(logsumexp(ll, axis=1))
    assert np.allclose(ll_fast, ll_slow), "fast doesn't match slow"
    assert np.allclose(ll_faster, ll_fast), "faster doesn't match faster"
    #assert np.allclose(ll_faster, np.log(ll_fastest)), "faster doesn't match cython fastest"
    print "GMM LIKE TESTS PASS"

def time_gmm_like():
    print "=== Timing GMM likes for speed ==="
    setup = """
import numpy as np
from CelestePy.util.like import gmm_like_2d, gmm_like
np.random.seed(41)
K = 42
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
    """

    cython_string = """
ll_faster = np.zeros(X.shape[0])
gmm_like_2d(ll_faster, x=X.astype(np.float), 
            ws=ws.astype(np.float),
            mus=means.astype(np.float), sigs=covs.astype(np.float))
"""

    num_reps = 30
    gmm_cython = timeit.timeit(cython_string, setup, number = num_reps) / num_reps
    gmm_numpy  = timeit.timeit("gmm_like(X, ws, means, covs)", setup, number=num_reps) / num_reps
    print "Cython Time: %2.4f per call"%gmm_cython
    print "Numpy  Time: %2.4f per call"%gmm_numpy
    print "Cython is %2.2f times faster"%(gmm_numpy / gmm_cython) 
    print ""

if __name__ == "__main__":
    #test_multivariate_normal_log_pdf()
    test_gmm_like()
    time_gmm_like()
