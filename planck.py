# 
# Functions for computing expected number of photons for a given
# temperature value for a given band
#
from collections import defaultdict
import numpy as np
import os

#######################################################################
# constants
#######################################################################
h = 6.6260693e-34  # Planck constant (Joules * second)
k = 1.3806488e-23  # Boltzmann constant (Joules / Kelvin)
c = 299792458      # speed of light (meters / second)
hc_k   = (h * c) / k
hpicc2 = 2 * np.pi * h * c * c

#Stefan-Boltzmann constant
sigma       = 2 * np.pi**5 * k**4 / (15 * c**2 * h**3)  # J / (s * m^2 * K^4)
sun_wattage = 3.846 * 1e26    # Luminosity of the sun (Joules / Second)
sun_radius  = 6.995e8         # Radius of the sun (meters)
m_per_ly    = c * 31556952.   # meters in a light year
lens_area   = .75 * np.pi * 1.25**2 # in meters^2
exposure_duration = 54.
bands       = ['u', 'g', 'r', 'i', 'z']

#######################################################################
# Load wavelength and sensitivity curves for each band
#######################################################################
def load_filter_curves():
    filter_file = os.path.dirname(os.path.realpath(__file__)) + "/" + "filter_curves"
    filter_curves      = open(filter_file, 'r')
    wavelength_lookup  = defaultdict(list)
    sensitivity_lookup = defaultdict(list)
    for line in filter_curves: 
        band, wavelength, sensitivity = line.strip().split('\t')
        wavelength_lookup[band].append(float(wavelength)*1e-4*1e-6) # angs to meters
        sensitivity_lookup[band].append(float(sensitivity))
    filter_curves.close()
    for b in bands: 
        wavelength_lookup[b] = np.array(wavelength_lookup[b])
        sensitivity_lookup[b] = np.array(sensitivity_lookup[b])
    return wavelength_lookup, sensitivity_lookup
wavelength_lookup, sensitivity_lookup = load_filter_curves()

#######################################################################
# Expected number of photons computation
#######################################################################
def photons_per_joule(T, band):
    """ Computes the number of expected photons for a given band at a given 
        energy level?? 
        Input: 
            - T
            - band: string in bands
    """
    x = wavelength_lookup[band]
    if len(x) == 0: 
      raise Exception('Band band: %s'%band)

    # compute radiance density (watts/m^2) as a function of temperature and band
    radiances          = hpicc2 / (np.power(x, 5.) * (np.exp(hc_k / (x * T)) - 1))
    total_radiance     = sigma * np.power(T, 4.) #across bands, per m^2
    radiance_densities = radiances / total_radiance

    photon_energies = (h * c / x) # Photon energy for each wavelength (in Joules)
    photon_fluxes   = radiance_densities / photon_energies
    filtered_photon_fluxes = photon_fluxes * sensitivity_lookup[band]

    avg_photons = np.mean(filtered_photon_fluxes) #per hertz
    range       = np.abs(x[2] - x[1]) * len(x)
    return avg_photons * range #approximates the integral

def photons_expected(T, solar_L, d, band):
    L          = solar_L * sun_wattage          # Joules/Seconds of source
    D          = d * m_per_ly                   # Distance of source
    lens_prop  = lens_area / (4*np.pi * D**2)   # proportion if energy captured by lens at distance D
    lens_watts = lens_prop * L                  # Joules/Seconds of source => lens
    return photons_per_joule(T, band) * lens_watts * exposure_duration

# for testing
if __name__=="__main__":

  # sensitivity_lookup[1] = [1. for i in 1:1000000]
  # wavelength_lookup[1] = [1:1000000] * 1e-10

  # the sun
  T = 6000.
  d = 150e9
  # sun_radius^2 * 4pi
  print photons_per_joule(T, 'r') * sun_wattage
  print photons_expected(T, 1., d / m_per_ly, 'r') # 5e22 photons in the r band?

  # vega
  # surface_area(9602., 40.12 * sun_wattage)
  # pi * (196e7)^2
  print "%2.2e"%photons_expected(9602., 40.12, 25.04, 'r')  ### 6e11 photons in the r band?

  # arcturus
  print "%2.2e"%photons_expected(4290., 170., 36.7, 'r')  ### 1e12 photons in the r band?

  # a supergiant on the edge of the milky way?
  print "%2.2e"%hotons_expected(3602., 400000., 90e3) ### 4e8 photons in the r band?

  # our sun at the edge of the milky way?
  print "%2.2e"%photons_expected(6000., 1., 90e3) ### 1685 photons in the r band?






