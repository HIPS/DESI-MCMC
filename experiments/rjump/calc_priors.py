import scipy.stats as stats
import numpy as np
import fitsio
from sklearn import mixture

import pickle
import pandas as pd

def mags2nanomaggies(mags):
    return np.power(10., (mags - 22.5)/-2.5)

def df_from_fits(filename, i=1):
    """ create a pandas dataframe from a fits file """
    return pd.DataFrame.from_records(fitsio.FITS(filename)[i].read().byteswap().newbyteorder())

def fit_fluxes(fluxes, filename, k = 4):
    g = mixture.GMM(n_components=4, covariance_type='full')
    g.fit(fluxes)
    output = open(filename, 'wb')
    pickle.dump(g, output)
    output.close()

"""
re  - [0, infty], transformation log
ab  - [0, 1], transformation log (ab / (1 - ab))
phi - [0, 180], transformation log (phi / (180 - phi))
"""
def fit_gal_shape(re, ab, phi, filename, k = 4):
    gre = mixture.GMM(n_components=4, covariance_type='full')
    gre.fit(np.log(re))

    gab = mixture.GMM(n_components=4, covariance_type='full')
    gab.fit(np.log(ab / (1 - ab)))

    gphi = mixture.GMM(n_components=4, covariance_type='full')
    gphi.fit(np.log(phi / (180 - phi)))

    output = open(filename, 'wb')
    pickle.dump((gre, gab, gphi), output)
    output.close()


# read galaxy and star from FITS
print "reading in galaxy and star fluxes"
data_gals = fitsio.FITS('../../data/existing_catalogs/gals.fits')[1].read()
data_stars = fitsio.FITS('../../data/existing_catalogs/stars.fits')[1].read()

print "reading in co-added galaxies"
test_coadd_fn = "../../data/stripe_82_dataset/square_106_4.fit"
coadd_df = df_from_fits(test_coadd_fn)

# store as (log nanomaggies)
fluxes_gals = np.zeros((len(data_gals['cmodelmag_u']), 5))
fluxes_stars = np.zeros((len(data_stars['psfmag_u']), 5))

print "creating array of fluxes"
bands = ['u', 'g', 'r', 'i', 'z']
for bandn,band in enumerate(bands):
    galaxy_name = 'cmodelmag_' + band
    star_name = 'psfmag_' + band
    for index, r in enumerate(data_gals[galaxy_name]):
        fluxes_gals[index][bandn] = np.log(mags2nanomaggies(r))
    for index, r in enumerate(data_stars[star_name]):
        fluxes_stars[index][bandn] = np.log(mags2nanomaggies(r))

valid_fluxes_gals  = np.array([True] * len(data_gals['cmodelmag_u']))
valid_fluxes_stars = np.array([True] * len(data_stars['psfmag_u']))
for bandn,band in enumerate(bands):
    valid_fluxes_gals  = valid_fluxes_gals & (fluxes_gals[:,bandn] != np.inf)
    valid_fluxes_stars = valid_fluxes_stars & (fluxes_stars[:,bandn] != np.inf)

fluxes_gals_final  = fluxes_gals[valid_fluxes_gals,:]
fluxes_stars_final = fluxes_stars[valid_fluxes_stars,:]

print "fitting galaxy fluxes"
fit_fluxes(fluxes_gals_final[:10000,:], 'gal_fluxes_mog.pkl')

print "fitting star fluxes"
fit_fluxes(fluxes_stars_final[:10000,:], 'star_fluxes_mog.pkl')

print "fitting galaxy shape"
fit_gal_shape(coadd_df['expRad_r'] + 0.01,
              coadd_df['expAB_r'],
              coadd_df['expPhi_r'],
              'gal_shape_mog.pkl')

