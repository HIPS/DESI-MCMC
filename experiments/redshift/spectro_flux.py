import fitsio
import sys
sys.path.append("../..")
import planck
import numpy as np

# spec file has one full spectrum (and model spectrum and lots of other stuff)
#spec_data = fitsio.FITS('data/spec-3690-55182-0114.fits')
spec_data = fitsio.FITS('../../data/DR10QSO/specs/spec-3754-55488-0762.fits')
noisy_spectrum = spec_data[1]['flux'].read()
noisy_spectrum_ivar = spec_data[1]['ivar'].read()

lam = 10 ** (spec_data[1]['loglam'].read())

###########################################################
# integrate THIS against SDSS FILTERS and use conversions
model_spec = spec_data[1]['model'].read()  

# TO MATCH THIS
spectro_syn_flux = spec_data[2]['SPECTROSYNFLUX'].read()   #UGRIZ FLUXES

## using SDSS FILTERS
planck.sensitivity_lookup 
planck.wavelength_lookup

def find_closest(wavelength):
    return model_spec[(np.abs(lam - wavelength)).argmin()]


#def project_to_bands(spectra, wavelengths): 
#    fluxes = np.zeros(5)
#    for i, band in enumerate(['u','g','r','i','z']):
#        # interpolate sensitivity curve onto wavelengths
#        sensitivity = np.interp(wavelengths, planck.wavelength_lookup[band]*(10**10), 
#                                             planck.sensitivity_lookup[band])
#        norm        = sum(sensitivity)
#
#        # conversion
#        flambda2fnu  = wavelengths**2 / 2.99792e18
#        fthru        = np.sum(sensitivity * spectra * flambda2fnu) / norm #np.multiply(model_matched, flambda2fnu)) / norm 
#        mags         = -2.5 * np.log10(fthru) - (48.6 - 2.5*17)
#        fluxes[i]    = np.power(10., (mags - 22.5)/-2.5)
#    return fluxes

from redshift_utils import project_to_bands

print spectro_syn_flux
print project_to_bands(model_spec, lam)

# (other broadband flux fields (UGRIZ fluxes)
spectroflux = spec_data[2]['SPECTROFLUX'].read()
spectroflux_ivar = spec_data[2]['SPECTROFLUX_IVAR'].read()
psfflux = spec_data[2]['PSFFLUX'].read()

