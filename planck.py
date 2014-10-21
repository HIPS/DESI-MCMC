# 
# Functions for computing expected number of photons for a given
# temperature value for a given band
#
from collections import defaultdict
import numpy as np
from scipy.optimize import fmin
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
    h_vecs = {}
    for b in bands: 
        wavelength_lookup[b] = np.array(wavelength_lookup[b])
        sensitivity_lookup[b] = np.array(sensitivity_lookup[b])

        # number of photons/Joule that make it through the filter at each energy level
        dLam =  np.abs(wavelength_lookup[b][2] - wavelength_lookup[b][1])
        h_vecs[b] = sensitivity_lookup[b] / (h * c / wavelength_lookup[b]) * dLam
    return wavelength_lookup, sensitivity_lookup, h_vecs
wavelength_lookup, sensitivity_lookup, h_vecs = load_filter_curves()

def spec_density(T, wavelengths):
    """ returns the spectral density for temperature t evaluated at wavelengths """
    radiances          = hpicc2 / (np.power(wavelengths, 5.) * (np.exp(hc_k / (wavelengths * T)) - 1))
    total_radiance     = sigma * np.power(T, 4.) #across bands, per m^2
    radiance_densities = radiances / total_radiance
    return radiance_densities

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
    #radiances          = hpicc2 / (np.power(x, 5.) * (np.exp(hc_k / (x * T)) - 1))
    #total_radiance     = sigma * np.power(T, 4.) #across bands, per m^2
    #radiance_densities = radiances / total_radiance
    f_vec = spec_density(T, x)

    # filter and convert to photon counts
    return f_vec.dot(h_vecs[band])
    #photon_energies = (h * c / x) # Photon energy for each wavelength (in Joules)
    #photon_fluxes   = radiance_densities / photon_energies
    #filtered_photon_fluxes = photon_fluxes * sensitivity_lookup[band]

    #avg_photons = np.mean(filtered_photon_fluxes) #per hertz
    #range       = np.abs(x[2] - x[1]) * len(x)
    #return avg_photons * range #approximates the integral

def photons_expected(T, solar_L, d, band):
    #L          = solar_L * sun_wattage          # Joules/Seconds of source
    #D          = d * m_per_ly                   # Distance of source
    #lens_prop  = lens_area / (4*np.pi * D**2)   # proportion if energy captured by lens at distance D
    #lens_watts = lens_prop * L                  # Joules/Seconds of source => lens

    # compute brightness
    brightness = solar_L / (d**2)
    return photons_expected_brightness(T, brightness, band)
    #return photons_per_joule(T, band) * lens_watts * exposure_duration

def photons_expected_brightness(T, b, band):
    """ same as above, except this uses b = ell/d^2 """
    lens_prop  = lens_area / (4 * np.pi)         # lens solid angle
    lens_watts = lens_prop * b * sun_wattage / (m_per_ly**2) # Joules/Seconds of source => lens
    return photons_per_joule(T, band) * lens_watts * exposure_duration 

def plancks_law(t, W):
    """ Planck's law, defines spectral radiance 
        Returns in units of Watts / m^2 / sec / steradian
    """
    radiance = (2 * h * c * c) / (np.power(W, 5.) * (np.exp(hc_k/(W*t)) - 1))
    return radiance



def planck_regression(spec_obs, wavelengths, isd_error, t0=None, 
                      fix_temp=False, disp=False):
    """ Performs a two dimensional regression using planck's law. 
        Input:
          spec_obs    : spectral observations in J/s/m^2/ang
          wavelengths : corresponding wavelengths in angstroms
          isd_error   : inverse standard error (in J/s/m^2/ang)
          t0          : initial temperature
          fix_temp    : fix temperature to t0 if true
          disp        : print optimization output

        Output:
          t_fit       : temperature fit
          ell_fit     : luminosity fit
          fit_spec    : ideal planck's law curve corresponding to "wavelengths"
    """
    def loss_func(th):
        temp = th[0]
        lum  = th[1]
        radiance = np.exp(lum) * plancks_law(temp, wavelengths*1e-10)
        return np.sum(np.abs(spec_obs - radiance) * isd_error)

    # make sure the temperature makes sense
    if t0 is None:
        assert fix_temp == False, "Need to input a temp to fix the temp"
        t0 = 6000

    # Fix temperature and run optimization
    if not fix_temp:
        t_fit, ell_fit = fmin(loss_func, x0      = np.array([t0, -13]),
                                         maxiter = 1000,
                                         maxfun  = 1000,
                                         disp    = disp)
    else:
        ell_fit = fmin(lambda(lum): loss_func([t0, lum]), x0 = -13,
                                                          maxiter = 1000,
                                                          maxfun  = 1000, 
                                                          disp    = disp)
        t_fit   = t0

    # compute full curve and return
    fit_spec = np.exp(ell_fit) * plancks_law(t_fit, wavelengths*1e-10)
    return t_fit, ell_fit, fit_spec


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
  print "%2.2e"%photons_expected(3602., 400000., 90e3, 'r') ### 4e8 photons in the r band?

  # our sun at the edge of the milky way?
  print "%2.2e"%photons_expected(6000., 1., 90e3, 'r') ### 1685 photons in the r band?






