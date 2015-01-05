import numpy as np
import fitsio
import sys
sys.path.append("../..")
import planck
import scipy.integrate as integrate
from scipy import interpolate
from scipy.optimize import minimize
from funkyyak import grad, numpy_wrapper as np
import matplotlib.pyplot as plt

def sinc_interp(new_samples, samples, fvals, left=None, right=None):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html        
    """
    if len(fvals) != len(samples):
        raise Exception, 'function vals (fvals) and samples must be the same length'

    # Find the period  
    T = (samples[1:] - samples[:-1]).max()

    # sinc resample
    sincM = np.tile(new_samples, (len(samples), 1)) - \
            np.tile(samples[:, np.newaxis], (1, len(new_samples)))
    y = np.dot(fvals, np.sinc(sincM/T))

    # set outside values to left/right inputs if given
    if left is not None:
        y[new_samples < samples[0]] = np.nan
    if right is not None:
        y[new_samples > samples[-1]] = np.nan
    return y

def spline_interp(new_samples, samples, fvals):
    tck  = interpolate.splrep(samples, fvals, s=0)
    ynew = interpolate.splev(new_samples, tck, der=0)
    return ynew

def load_data_clean_split(spec_fits_file = '../../andrew-qso.fits', Ntrain=500):

    # load and split
    fits_data = fitsio.FITS(spec_fits_file)

    # compute wavelength values
    log10lams   = fits_data[0].read()
    wavelengths = np.power(10, log10lams)

    # load spectra samples
    quasar_spectra = fits_data[1].read()
    #quasar_spectra[quasar_spectra <= 0] = quasar_spectra[quasar_spectra > 0].min()
    quasar_ivar    = fits_data[2].read()

    # load known red shift
    meta_data = fits_data[3].read()
    quasar_z = meta_data['Z']
    quasar_zerr = meta_data['Z_ERR']

    # split train/test
    np.random.seed(42)
    perm = np.random.permutation(quasar_spectra.shape[0])
    train_idx = perm[0:Ntrain]
    test_idx  = perm[Ntrain:]

    trainObj = {}
    trainObj['spectra']      = quasar_spectra[train_idx, :]
    trainObj['spectra_ivar'] = quasar_ivar[train_idx, :]
    trainObj['Z']            = quasar_z[train_idx]
    trainObj['Z_err']        = quasar_zerr[train_idx]

    testObj = {}
    testObj['spectra']      = quasar_spectra[test_idx, :]
    testObj['spectra_ivar'] = quasar_ivar[test_idx, :]
    testObj['Z']            = quasar_z[test_idx]
    testObj['Z_err']        = quasar_zerr[test_idx]
    return wavelengths, trainObj, testObj

def project_to_bands(spectra, wavelengths):
    # linearly interpolate filters to be sampled the same as the spectra
    bands = ['u', 'g', 'r', 'i', 'z']
    filters = {}
    for b in bands: 
        filter_interp = np.interp(wavelengths,
                                  planck.wavelength_lookup[b] * 1e10,
                                  planck.sensitivity_lookup[b],
                                  left = 0, right = 0)
        filters[b] = filter_interp

    # numerically integrate filter*observed spectra to get band brightness 
    bright_mat = np.zeros((spectra.shape[0], len(bands)))
    for n in range(spectra.shape[0]):
        for i, b in enumerate(bands):
            bright_mat[n, i] = integrate.simps(spectra[n,:]*filters[b],
                                               wavelengths)
    return bright_mat


def fit_weights_given_basis(B, lam0, X, inv_var, z_n, lam_obs, return_loss=False, sgd_iter=100):
    """ Weighted optimization routine to fit the values of \log w given 
    basis B. 
    """
    #convert spec_n to lam0
    spec_n_resampled = np.interp(lam0, lam_obs/(1+z_n), X, left=np.nan, right=np.nan)
    ivar_n_resampled = np.interp(lam0, lam_obs/(1+z_n), inv_var, left=np.nan, right=np.nan)
    spec_n_resampled[np.isnan(spec_n_resampled)] = 0.0
    ivar_n_resampled[np.isnan(ivar_n_resampled)] = 0.0
    def loss_omegas(omegas):
        """ loss over weights with respect to fixed basis """
        ll_omega = .5 / (100.) * np.sum(np.square(omegas))
        Xtilde   = np.dot(np.exp(omegas), B)
        return np.sum(ivar_n_resampled * np.square(spec_n_resampled - Xtilde)) + ll_omega
    loss_omegas_grad = grad(loss_omegas)

    # first wail on it with gradient descent/momentum
    omegas        = .01*np.random.randn(B.shape[0])
    momentum      = .9
    learning_rate = 1e-4
    cur_dir = np.zeros(omegas.shape)
    lls     = np.zeros(sgd_iter)
    for epoch in range(sgd_iter):
        grad_th    = loss_omegas_grad(omegas)
        cur_dir    = momentum * cur_dir + (1.0 - momentum) * grad_th
        omegas    -= learning_rate * cur_dir
        lls[epoch] = loss_omegas(omegas)

        step_mag = np.sqrt(np.sum(np.square(learning_rate*cur_dir)))
        if epoch % 20 == 0:
            print "{0:15}|{1:15}|{2:15}".format(epoch, "%7g"%lls[epoch], "%2.4f"%step_mag)

    # tighten it up w/ LBFGS
    res = minimize(x0 = omegas,
                   fun = loss_omegas,
                   jac=loss_omegas_grad,
                   method = 'L-BFGS-B',
                   options = { 'disp': True, 'maxiter': 10000 })

    # return the loss function handle as well - for debugging
    if return_loss:
        return np.exp(res.x), loss_omegas
    return np.exp(res.x)

def evaluate_random_direction(fun, x0, n=100, delta=.1):
    """ plots a multivariate function over one (random) direction """
    # random direciton w/ magnitude delta
    param_scale = .1
    rand_dir = np.random.randn(x0.size) * param_scale
    rand_dir = delta * rand_dir / np.sqrt(np.dot(rand_dir, rand_dir))
    # bounds
    x_left  = x0 - n*rand_dir
    ll_grid = np.zeros(2*n+1)
    x = x_left
    for n in range(len(ll_grid)):
        x = x_left + n*rand_dir
        ll_grid[n] = fun(x)
    return ll_grid

def check_grad(fun, jac, th):
    """ check the gradient along a random direction """
    param_scale = .1
    rand_dir    = np.random.randn(th.size) * param_scale
    rand_dir    = rand_dir / np.sqrt(np.dot(rand_dir, rand_dir))
    test_fun    = lambda x : fun(th + x * rand_dir.reshape(th.shape))
    nd          = (test_fun(1e-4) - test_fun(-1e-4)) / 2e-4
    ad          = np.dot(jac(th).ravel(), rand_dir)
    print "Checking grads. Relative diff is: {0}".format((nd - ad)/np.abs(nd))


