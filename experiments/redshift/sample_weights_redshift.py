import fitsio
import numpy as np
import numpy.random as npr
from scipy.optimize import minimize
import sys, os
from redshift_utils import project_to_bands
from quasar_fit_basis import load_basis_fit

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

def prior_w(w): 
    if np.any(w <= 0): 
        return -np.inf
    return 0

def prior_z(z): 
    if np.any(z <= 0): 
        return -np.inf
    return 0

if __name__=="__main__":

    ## load ML basis from cache (beta and omega values)
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

    # Draw posterior samples p(w, z, m | B, y_flux, y_flux_ivar)
    Nsamps = 1000
    w_samps = np.zeros((Nsamps, B.shape[0]))  # N samples of a K dimensional vector
    z_samps = np.zeros(Nsamps)                # N samples of a scalar
    m_samps = np.zeros(Nsamps)                # N samples of the magnitude



