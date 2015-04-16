import scipy.stats as stats
import numpy as np
import fitsio
from sklearn import mixture

# read from FITS file
data_gals = fitsio.FITS('data_for_priors/gals.fits')[1].read()
data_stars = fitsio.FITS('data_for_priors/stars.fits')[1].read()

bands_gals = np.zeros((len(data_gals['cmodelmag_u']), 5))
bands_stars = np.zeros((len(data_stars['psfmag_u']), 5))

print "creating np arrays of data"
bands = ['u', 'g', 'r', 'i', 'z']
for bandn,band in enumerate(bands):
    galaxy_name = 'cmodelmag_' + band
    star_name = 'psfmag_' + band
    for index, r in enumerate(data_gals[galaxy_name]):
        bands_gals[index][bandn] = r
    for index, r in enumerate(data_stars[star_name]):
        bands_stars[index][bandn] = r

print "creating relative data"
relative_bands_gals = np.zeros((len(data_gals['cmodelmag_u']), 4))
relative_bands_stars = np.zeros((len(data_stars['psfmag_u']), 4))

for i in range(len(relative_bands_gals)):
    for bandn,band in enumerate(['u', 'g', 'i', 'z']):
        if band == 'i' or band == 'z':
            relative_bands_gals[i][bandn] = bands_gals[i][bandn+1] / bands_gals[i][2]
        else:
            relative_bands_gals[i][bandn] = bands_gals[i][bandn] / bands_gals[i][2]

for i in range(len(relative_bands_stars)):
    for bandn,band in enumerate(['u', 'g', 'i', 'z']):
        if band == 'i' or band == 'z':
            relative_bands_stars[i][bandn] = bands_stars[i][bandn+1] / bands_stars[i][2]
        else:
            relative_bands_stars[i][bandn] = bands_stars[i][bandn] / bands_stars[i][2]

### FIRST TYPE OF FIT
# fit an independent gamma to each band
print "trying independent gammas"
gammas = []
len_gals = len(bands_gals)
train_gals = bands_gals[:int(0.99 * len_gals), :]
validation_gals = bands_gals[int(0.99 * len_gals):, :]

len_stars = len(bands_stars)
train_stars = bands_stars[:int(0.99 * len_stars), :]
validation_stars = bands_stars[int(0.99 * len_stars):, :]

gammas_galaxies = []
gammas_stars = []
for i in range(len(bands)):
    print "fitting gamma for band", bands[i]
    gammas_galaxies.append(stats.gamma.fit(train_gals[:,i]))
    gammas_stars.append(stats.gamma.fit(train_stars[:,i]))

print "Galaxies:", gammas_galaxies
print "Stars:", gammas_stars

gammas_gal_vals = 0
gammas_stars_vals = 0
for i in range(len(bands)):
    print "validation gamma for band", bands[i]
    (a0, loc0, scale0) = gammas_galaxies[i]
    (a1, loc1, scale1) = gammas_stars[i]

    gammas_gal_vals += sum(stats.gamma.logpdf(validation_gals[:,i], a0, loc0, scale0))
    gammas_stars_vals += sum(stats.gamma.logpdf(validation_stars[:,i], a1, loc1, scale1))

print "log likelihood for galaxies:", gammas_gal_vals
print "log likelihood for stars:", gammas_stars_vals


### SECOND TYPE OF FIT
# fit middle band to gamma
# fit a multivariate normal to log ratios of other bands over that band 
print "trying relative mixture of normal + gamma"
gammas = []
len_gals = len(bands_gals)
train_relative_gals = relative_bands_gals[:int(0.99 * len_gals),:]
validation_relative_gals = relative_bands_gals[int(0.99 * len_gals):,:]

len_stars = len(bands_stars)
train_relative_stars = relative_bands_stars[:int(0.99 * len_stars),:]
validation_relative_stars = relative_bands_stars[int(0.99 * len_stars):,:]

(a_main_gal, loc_main_gal, scale_main_gal) = stats.gamma.fit(train_gals[:,2])
(a_main_star, loc_main_star, scale_main_star) = stats.gamma.fit(train_stars[:,2])

print "gamma on galaxies r:", a_main_gal, loc_main_gal, scale_main_gal
print "gamma on stars r:", a_main_star, loc_main_star, scale_main_star

g_gals = mixture.GMM(n_components=4, covariance_type='full')
g_gals.fit(train_relative_gals)
score_gals = g_gals.score(validation_relative_gals)
print (sum(score_gals) +
        sum(stats.gamma.logpdf(validation_gals[:,2], a_main_gal, loc_main_gal, scale_main_gal)))

print "galaxy GMM:"
print g_gals.weights_
print g_gals.means_
print g_gals.covars_


g_stars = mixture.GMM(n_components=4, covariance_type='full')
g_stars.fit(train_relative_stars)
score_stars = g_stars.score(validation_relative_stars)
print (sum(score_gals) +
        sum(stats.gamma.logpdf(validation_stars[:,2], a_main_star, loc_main_star, scale_main_star)))

print "star GMM:"
print g_stars.weights_
print g_stars.means_
print g_stars.covars_
