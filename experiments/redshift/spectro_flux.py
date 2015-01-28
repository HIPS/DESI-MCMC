import fitsio
import sys
sys.path.append("../..")
import planck

# spec file has one full spectrum (and model spectrum and lots of other stuff)
spec_data = fitsio.FITS('../../data/DR10QSO/specs/spec-6063-56098-0578.fits')
noisy_spectrum = spec_data[1]['flux'].read()
noisy_spectrum_ivar = spec_data[1]['ivar'].read()

###########################################################
# integrate THIS against SDSS FILTERS and use conversions
model_spec = spec_data[1]['model'].read()  

# TO MATCH THIS
spectro_syn_flux = spec_data[2]['SPECTROSYNFLUX'].read()   #UGRIZ FLUXES

## using SDSS FILTERS
planck.sensitivity_lookup 
planck.wavelength_lookup

# (other broadband flux fields (UGRIZ fluxes)
spectroflux = spec_data[2]['SPECTROFLUX'].read()
spectroflux_ivar = spec_data[2]['SPECTROFLUX_IVAR'].read()
psfflux = spec_data[2]['PSFFLUX'].read()

