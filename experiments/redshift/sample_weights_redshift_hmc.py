import fitsio
import numpy as np
import numpy.random as npr
from scipy.optimize import minimize
import sys, os
from redshift_utils import project_to_bands
from quasar_fit_basis import load_basis_fit

### helper functions
def softmax(x):
    return np.exp(x) / sum(np.exp(x))

# TODO: use real priors
def prior_w(w): 
    if np.any(w <= 0): 
        return -np.inf
    return 0

def prior_z(z): 
    if np.any(z <= 0): 
        return -np.inf
    return 0

def prior_m(m):
    if np.any(m <= 0):
        return -np.inf
    return 0

### Likelihood of 5-band SDSS flux given weights, 
def pixel_likelihood(z, w, m, fluxes, fluxes_ivar, lam0, B):
    """ compute the likelihood of 5 bands given
        z    : (scalar) red-shift of observed source
        w    : (vector) K positive weights for positive rest-frame basis
        x    : (vector) 5 pixel values corresponding to UGRIZ
        lam0 : basis wavelength values
        B    : (matrix) K x P basis 
    """
    # at rest frame for lam0
    lam_obs = lam0 * (1. + z)
    spec    = w.dot(B)
    mu      = project_to_bands(spec, lam_obs)
    ll      = -.5 * mu * np.sum(x * np.log(mu) - mu)
    return ll

### Let gamma be a vector whose softmax gives weights that sum to 1
def grad_pixel_likelihood(z, gamma, m, fluxes, fluxes_ivar, lam0, B,
                          delta_z, delta_gamma, delta_m):

    lls = []
    w = softmax(gamma)

    # take gradient with respect to z
    z_lower -= delta_z
    z_upper += delta_z
    l_lower = pixel_likelihood(z_lower, w, m, fluxes, fluxes_ivar, lam0, B)
    p_lower = l_lower * prior_z(z_lower) * prior_w(w) * prior_m(m)

    l_upper = pixel_likelihood(z_upper, w, m, fluxes, fluxes_ivar, lam0, B)
    p_upper = l_upper * prior_z(z_upper) * prior_w(w) * prior_m(m)

    grad_z = (p_upper - p_lower) / (2 * delta_z)
    lls.append(grad_z)

    # take gradient with respect to gamma
    for i in range(len(gamma)):
        gamma_lower = deepcopy(gamma)
        gamma_upper = deepcopy(gamma)
        gamma_lower[i] -= delta_gamma
        gamma_upper[i] += delta_gamma
        w_lower = softmax(gamma_lower)
        w_upper = softmax(gamma_upper)
        l_lower = pixel_likelihood(z, w_lower, m, fluxes, fluxes_ivar, lam0, B)
        p_lower = l_lower * prior_z(z) * prior_w(w_lower) * prior_m(m)

        l_upper = pixel_likelihood(z, w_upper, m, fluxes, fluxes_ivar, lam0, B)
        p_upper = l_upper * prior_z(z) * prior_w(w_upper) * prior_m(m)

        grad_gamma = (p_upper - p_lower) / (2 * delta_gamma)
        lls.append(grad_gamma)

    # take gradient with respect to m
    m_lower = m - delta_m
    m_upper = m + delta_m
    l_lower = pixel_likelihood(z, w, m_lower, fluxes, fluxes_ivar, lam0, B)
    p_lower = l_lower * prior_z(z) * prior_w(w) * prior_m(m_lower)

    l_upper = pixel_likelihood(z, w, m_upper, fluxes, fluxes_ivar, lam0, B)
    p_upper = l_upper * prior_z(z) * prior_w(w_lower) * prior_m(m_upper)

    grad_m = (p_upper - p_lower) / (2 * delta_m)
    lls.append(grad_m)

    return np.array(lls)


### load ML basis from cache (beta and omega values)
basis_cache = 'cache/basis_fit_K-4_V-1364.pkl'
USE_CACHE = True
if os.path.exists(basis_cache) and USE_CACHE:
    th, lam0, lam0_delta, parser = load_basis_fit(basis_cache)

# compute actual weights and basis values (normalized basis + weights)
mus    = parser.get(th, 'mus')
betas  = parser.get(th, 'betas')
omegas = parser.get(th, 'omegas')
W = np.exp(omegas)
W = W / np.sum(W, axis=1, keepdims=True)
B = np.exp(betas)
B = B / np.sum(B * lam0_delta, axis=1, keepdims=True)
M = np.exp(mus)

# load in some quasar fluxes
qso_df        = fitsio.FITS('../../data/DR10QSO/DR10Q_v2.fits')
psf_flux      = qso_df[1]['PSFFLUX'].read()
psf_flux_ivar = qso_df[1]['IVAR_PSFFLUX'].read()
z             = qso_df[1]['Z_VI'].read()
print "Loaded %d spectroscopically measured quasars and their redshifts"%len(psf_flux)

# choose one to work on
n = 23
y_flux      = psf_flux[n]
y_flux_ivar = psf_flux_ivar[n]
print "Choosing n = %d, (z = %2.2f)"%(n, z[n])

# functions to pass into HMC
SIZE_BASIS = 4
def probability(q):
    m = q[0]
    w = q[1:SIZE_BASIS]
    r = q[SIZE_BASIS + 1]
    return pixel_likelihood(z, w, m, y_flux, y_flux_ivar, lam0, B)

def grad_probability(q):
    m = q[0]
    w = q[1:SIZE_BASIS]
    r = q[SIZE_BASIS + 1]
    return grad_pixel_likelihood(z, w, m, y_flux, y_flux_ivar, lam0, B):

if __name__=="__main__":
    # Draw posterior samples p(w, z, m | B, y_flux, y_flux_ivar)
    Nsamps = 1000
    w_samps = np.zeros((Nsamps, B.shape[0]))  # N samples of a K dimensional vector
    z_samps = np.zeros(Nsamps)                # N samples of a scalar
    m_samps = np.zeros(Nsamps)                # N samples of the magnitude

    # TODO: change samples 
    samps = np.zeros(Nsamps, B.shape[0] + 2)
    for s in np.arange(1, Nsamps)
        smpls[s] = hmc(probablility,
                       grad_probability,
                       0.1, 10,
                       np.atleast_1d(smpls[s-1]),
                       negative_log_prob=True)

