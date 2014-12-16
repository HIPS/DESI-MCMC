import numpy as np
import fitsio
import sys
sys.path.append("../..")
import planck
import scipy.integrate as integrate
from scipy import interpolate

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


