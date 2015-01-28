import fitsio
import sys
sys.path.append("../..")
import planck
import numpy as np

# spec file has one full spectrum (and model spectrum and lots of other stuff)
spec_data = fitsio.FITS('data/spec-3690-55182-0114.fits')
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

fluxes = dict()

for band in ['u','g','r','i','z']:
    wavelengths = planck.wavelength_lookup[band] * (10**10)
    sensitivity = planck.sensitivity_lookup[band]

    model_matched = np.array(map(find_closest, wavelengths))
    fthru = np.dot(sensitivity, model_matched)
    fluxes[band] = -2.5 * np.log10(fthru) - (48.6 - 2.5*17)

print fluxes

# (other broadband flux fields (UGRIZ fluxes)
spectroflux = spec_data[2]['SPECTROFLUX'].read()
spectroflux_ivar = spec_data[2]['SPECTROFLUX_IVAR'].read()
psfflux = spec_data[2]['PSFFLUX'].read()

