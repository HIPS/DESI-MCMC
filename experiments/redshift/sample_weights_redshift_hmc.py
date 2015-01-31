import fitsio
import numpy as np
import numpy.random as npr
from scipy.optimize import minimize
from hmc import hmc
import sys, os
from redshift_utils import project_to_bands
from quasar_fit_basis import load_basis_fit
import scipy.stats
import matplotlib.pyplot as plt
import copy

### helper functions
def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def prior_omega(omega):
    return -.5 * omega.dot(omega)

# TODO(awu): Gaussian for now
MEAN_Z_DIST = 2.5
STDEV_Z_DIST = 1.0
def prior_z(z): 
    return -(z - MEAN_Z_DIST)*(z- MEAN_Z_DIST) / (2. * STDEV_Z_DIST * STDEV_Z_DIST)

# TODO(awu): lognormal for now
STDEV_M_DIST = 20.0
def prior_mu(mu):
    return - mu * mu / (2. * STDEV_M_DIST*STDEV_M_DIST)

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
    spec    = np.dot(w, B)
    mu      = project_to_bands(spec, lam_obs) * m / (1. + z)
    ll      = -0.5 * np.sum(np.multiply(fluxes_ivar, (fluxes - mu)**2))
    return ll

### Let gamma be a vector whose softmax gives weights that sum to 1
def grad_probability_params(z, gamma, m, fluxes, fluxes_ivar, lam0, B,
                            delta_z, delta_gamma, delta_m):
    lls = []
    w = softmax(gamma)

    # take gradient with respect to z
    z_lower = z - delta_z
    z_upper = z + delta_z
    l_lower = pixel_likelihood(z_lower, w, m, fluxes, fluxes_ivar, lam0, B)
    p_lower = l_lower * prior_z(z_lower) * prior_gamma(gamma) * prior_m(m)

    l_upper = pixel_likelihood(z_upper, w, m, fluxes, fluxes_ivar, lam0, B)
    p_upper = l_upper * prior_z(z_upper) * prior_gamma(gamma) * prior_m(m)

    grad_z = (p_upper - p_lower) / (2 * delta_z)
    lls.append(grad_z)

    # take gradient with respect to gamma
    for i in range(len(gamma)):
        gamma_lower = copy.deepcopy(gamma)
        gamma_upper = copy.deepcopy(gamma)
        gamma_lower[i] -= delta_gamma
        gamma_upper[i] += delta_gamma
        w_lower = softmax(gamma_lower)
        w_upper = softmax(gamma_upper)
        l_lower = pixel_likelihood(z, w_lower, m, fluxes, fluxes_ivar, lam0, B)
        p_lower = l_lower * prior_z(z) * prior_gamma(gamma_lower) * prior_m(m)

        l_upper = pixel_likelihood(z, w_upper, m, fluxes, fluxes_ivar, lam0, B)
        p_upper = l_upper * prior_z(z) * prior_gamma(gamma_upper) * prior_m(m)

        grad_gamma = (p_upper - p_lower) / (2 * delta_gamma)
        lls.append(grad_gamma)

    # take gradient with respect to m
    m_lower = m - delta_m
    m_upper = m + delta_m
    l_lower = pixel_likelihood(z, w, m_lower, fluxes, fluxes_ivar, lam0, B)
    p_lower = l_lower * prior_z(z) * prior_gamma(gamma) * prior_m(m_lower)

    l_upper = pixel_likelihood(z, w, m_upper, fluxes, fluxes_ivar, lam0, B)
    p_upper = l_upper * prior_z(z) * prior_gamma(gamma) * prior_m(m_upper)

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
z_n = z[n]
y_flux      = psf_flux[n]
y_flux_ivar = psf_flux_ivar[n]
print "Choosing n = %d, (z = %2.2f)"%(n, z[n])

# functions to pass into HMC
def lnpdf(q):
    z     = q[0]
    omega = q[1:(B.shape[0] + 1)]
    mu    = q[B.shape[0] + 1]
    ll    =  pixel_likelihood(z, softmax(omega), np.exp(mu), y_flux, y_flux_ivar, lam0, B)
    return ll + prior_omega(omega) + prior_mu(mu) + prior_z(z)

# TODO(awu): set appropriately
INIT_REDSHIFT = 4.0
INIT_MAG = 10000.
STEP_SIZE = 0.00001
STEPS_PER_SAMPLE = 10

DELTA_Z = 0.01
DELTA_W = 0.01
DELTA_M = 1

def grad_lnpdf(q):
    z = q[0]
    gamma = q[1:(B.shape[0] + 1)]
    m = q[B.shape[0] + 1]
    return grad_probability_params(z, gamma, m, y_flux, y_flux_ivar, lam0, B,
                                   DELTA_Z, DELTA_W, DELTA_M)

# Metropolis-Hastings
def create_transition():
    mean = np.zeros(B.shape[0] + 2)
    var = np.zeros(B.shape[0] + 2)
    var[0] = 0.5
    var[1:(B.shape[0] + 1)] = 0.1
    var[B.shape[0] + 1] = 1000
    return np.random.multivariate_normal(mean, np.diag(var))

if __name__ == "__main__":
    # Draw posterior samples p(w, z, m | B, y_flux, y_flux_ivar)
    Nsamps = 5000
    w_samps = np.zeros((Nsamps, B.shape[0]))  # N samples of a K dimensional vector
    z_samps = np.zeros(Nsamps)                # N samples of a scalar
    m_samps = np.zeros(Nsamps)                # N samples of the magnitude
    likelihood_samps = np.zeros(Nsamps)

    samps = np.zeros((Nsamps, B.shape[0] + 2))
    samps[0,:] = np.zeros(B.shape[0] + 2)
    samps[0, 0] = INIT_REDSHIFT
    samps[0, B.shape[0] + 1] = np.log(INIT_MAG)
    Naccept = 0
    prop_scale = .05 * np.ones(samps.shape[1])
    prop_scale[-1] = .2
    for s in np.arange(1, Nsamps):

        if s > 2000: 
            prop_scale = .01 * np.ones(samps.shape[1])
            prop_scale[-1] = .1

        probability_pre  = lnpdf(samps[s-1,:])
        change           = create_transition()
        change           = prop_scale * np.random.randn(samps.shape[1])
        new_samp         = samps[s-1,:] + change
        probability_post = lnpdf(new_samp)

        # accept/reject
        if np.log(npr.rand()) < probability_post - probability_pre:
            samps[s,:]          = new_samp
            likelihood_samps[s] = probability_post
            Naccept += 1
        else:
            samps[s,:]          = samps[s-1,:]
            likelihood_samps[s] = probability_pre

        if s % 100==0:
            print "Iteration", s
            print "num accept (frac): %d (%2.2f)"%(Naccept, Naccept / float(s))
            print "z:", samps[s, 0]
            print "change: ", change
            print "w:", softmax(samps[s, 1:(B.shape[0] + 1)])
            print "m:", samps[s, B.shape[0] + 1]
            print "prob:", likelihood_samps[s]

    z_samps = samps[:,0]
    w_samps = samps[:,1:(B.shape[0] + 1)]
    m_samps = samps[:,B.shape[0] + 1]

    # plot z
    plt.figure(1)
    plt.plot(range(0, len(z_samps)), z_samps, 'bo')
    plt.savefig('z.png')

    # plot m
    plt.figure(2)
    plt.plot(range(0, len(m_samps)), m_samps, 'bo')
    plt.savefig('m.png')

    # plot likelihoods
    plt.figure(3)
    plt.plot(range(0, len(likelihood_samps)), likelihood_samps, 'bo')
    plt.savefig('ll.png')


    n, bins, patches = plt.hist(z_samps[2500:], 40, normed=True)
    plt.vlines(z_n, 0, n.max())
    plt.show()

"""
if __name__ == "__main__":
    # Draw posterior samples p(w, z, m | B, y_flux, y_flux_ivar)
    Nsamps = 10
    w_samps = np.zeros((Nsamps, B.shape[0]))  # N samples of a K dimensional vector
    z_samps = np.zeros(Nsamps)                # N samples of a scalar
    m_samps = np.zeros(Nsamps)                # N samples of the magnitude
    likelihood_samps = np.zeros(Nsamps)

    samps = np.zeros((Nsamps, B.shape[0] + 2))
    samps[0,:] = np.ones(B.shape[0] + 2)
    samps[0, 0] = INIT_REDSHIFT
    samps[0, B.shape[0] + 1] = INIT_MAG
    for s in np.arange(1, Nsamps):
        print "Iteration", s
        samps[s] = hmc(probability,
                       grad_probability,
                       STEP_SIZE, STEPS_PER_SAMPLE,
                       samps[s-1,:],
                       negative_log_prob=True)
        likelihood_samps[s] = probability(samps[s])
        print "z:", samps[s, 0]
        print "w:", softmax(samps[s, 1:(B.shape[0] + 1)])
        print "m:", samps[s, B.shape[0] + 1]

    z_samps = samps[:,0]
    w_samps = samps[:,1:(B.shape[0] + 1)]
    m_samps = samps[:,B.shape[0] + 1]

    # plot z
    plt.figure(1)
    plt.plot(range(0, len(z_samps)), z_samps, 'bo')
    plt.savefig('z.png')

    # plot m
    plt.figure(2)
    plt.plot(range(0, len(m_samps)), m_samps, 'bo')
    plt.savefig('m.png')

    # plot likelihoods
    plt.figure(3)
    plt.plot(range(0, len(likelihood_samps)), likelihood_samps, 'bo')
    plt.savefig('ll.png')
"""
