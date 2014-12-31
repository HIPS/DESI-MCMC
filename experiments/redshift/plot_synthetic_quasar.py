import numpy as np
import fitsio
import sys
sys.path.append("../..")
import planck
from scipy import interpolate
from celeste import FitsImage, celeste_likelihood_multi_image, gen_model_image
from util.init_utils import load_imgs_and_catalog
import numpy as np
import matplotlib.pyplot as plt
from redshift_utils import load_data_clean_split, project_to_bands
from glob import glob
import scipy.integrate as integrate

## grab some plotting defaults
import seaborn as sns
sns.set_style("white")
current_palette = sns.color_palette()

### set output directory ####################################################
out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"

## load a handful of quasar spectra
lam_obs, qtrain, qtest = \
    load_data_clean_split(spec_fits_file = 'quasar_data.fits', 
                          Ntrain = 400)

## first find a positive decomposition of quasar spectra on training data
quasar_spectra = qtrain['spectra']
quasar_z       = qtrain['Z']
quasar_ivar    = qtrain['spectra_ivar']
quasar_zerr    = qtrain['Z_err']
N              = quasar_spectra.shape[0]

## define a set of quasars to look at, and grab their pixel values
idxs = [0, 1, 4, 8]
for i, n in enumerate([0]): 
    mu_n        = project_to_bands(np.atleast_2d(quasar_spectra[n, :]), lam_obs).ravel()
    #x_n         = npr.poisson(mu_n).ravel()

    ## Generate some fake sources using real image data (PSF and stuff)
    cat_glob = glob('../../data/stamp_catalog/cat*.fits')[0:1]
    srcs, imgs, teff_catalog, us = load_imgs_and_catalog(cat_glob)

    ## create quasar source with known mean values for each band
    np.random.seed(42)
    srcs = srcs[1:]
    srcs[0].t = None
    srcs[0].b = None
    srcs[0].fluxes = dict(zip(['u', 'g', 'r', 'i', 'z'], mu_n))

    # re-generate images using these source params
    for img in imgs: 
        mimg      = gen_model_image(srcs, img)
        img.nelec = np.random.poisson(mimg)
        plt.imshow(img.nelec.T, interpolation='none')
        plt.savefig(out_dir + "img_%d_band_%s.pdf"%(n, img.band), bbox_inches='tight')


## plot the raw fluxes for each band
mus = project_to_bands(quasar_spectra[idxs, :], lam_obs.ravel())
mus /= mus.sum(axis=1, keepdims=True)
fig = plt.figure()
plt.imshow(mus, interpolation='nearest')
plt.xlabel('image band')
plt.ylabel('quasar index')
plt.title('Quasar flux distributions' )
plt.xticks(np.arange(mus.shape[1]), ['u', 'g', 'r', 'i', 'z'], fontsize=10)
plt.yticks(np.arange(mus.shape[0]), idxs)
plt.colorbar()
plt.savefig(out_dir + "quasar_sdss_fluxes.pdf", bbox_inches = 'tight', dpi=400)

## plot example spectroscopy 
fig = plt.figure(figsize=(18, 6))
plt.plot(lam_obs, quasar_ivar[0, :].T, alpha=.5, color="grey", label="inverse variance")
plt.plot(lam_obs, quasar_spectra[0, :].T, label="spectra", linewidth=2)
plt.ylim(0, quasar_spectra[0, :].max())
plt.legend(fontsize='xx-large')
plt.title("Quasar full spectrum")
plt.xlabel("wavelength")
plt.ylabel("$f(\lambda)$")
plt.savefig(out_dir + "quasar_spectrum_ivar.pdf", bbox_inches = 'tight')

## plot multiple and then plot multiple shifted
fig = plt.figure(figsize=(18, 6))
for idx in idxs:
    plt.plot(lam_obs, quasar_spectra[idx, :].T, label="$z = %2.2f$"%quasar_z[idx])
plt.ylim(0, quasar_spectra[idxs, :].max())
plt.legend(fontsize='xx-large')
plt.title("Red-shift comparison of quasar spectra")
plt.xlabel("wavelength")
plt.ylabel("$f(\lambda)$")
plt.savefig(out_dir + "quasar_redshift_obs_frame.pdf", bbox_inches = 'tight')


## plot a bunch in it's own directory
full_out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/quasar_specs/" 
full_idx = np.arange(0, qtrain['spectra'].shape[0], 5)
for idx in full_idx: 
    fig = plt.figure(figsize=(24,8))
    plt.plot(lam_obs, qtrain['spectra'][idx, :].T, label="z = %2.2f"%qtrain['Z'][idx], linewidth=1)
    plt.plot(lam_obs, qtrain['spectra_ivar'][idx, :].T, label="inv var", color='grey', alpha=.5, linewidth=.3)
    plt.title("Quasar (training # %d)"%idx)
    plt.xlabel("wavelength")
    plt.ylabel("$f(\lambda)$")
    plt.legend()
    plt.savefig(full_out_dir + "quasar_spectra_%d.png"%idx, bbox_inches = 'tight')
    plt.close('all')
 
## re-sample into rest frame
#header      = fitsio.read_header('../../data/eigen_specs/spEigenQSO-55732.fits')
#eigQSOfits  = fitsio.FITS('../../data/eigen_specs/spEigenQSO-55732.fits')
#lam0        = 10.**(header['COEFF0'] + np.arange(header['NAXIS1']) * header['COEFF1'])
##lam0        = lam0[::15]
#eigQSO      = eigQSOfits[0].read()[:,::15]
#K           = eigQSO.shape[0]

## resample to lam0 => rest frame basis 
#print "resampling de-redshifted data"
#wave_mat          = np.zeros(quasar_spectra.shape)
#spectra_resampled = np.zeros((quasar_spectra.shape[0], len(lam0)))
#spectra_ivar_resampled = np.zeros((quasar_spectra.shape[0], len(lam0)))
#for i in range(quasar_spectra.shape[0]):
#    wave_mat[i, :] = lam_obs / (1 + quasar_z[i])
#    spectra_resampled[i, :] = np.interp(x     = lam0,
#                                        xp    = wave_mat[i, :],
#                                        fp    = quasar_spectra[i, :],
#                                        left  = np.nan,
#                                        right = np.nan)
#    #spectra_ivar_resampled[i, :] = np.interp(x     = lam0,
#    #                                         xp    = wave_mat[i, :],
#    #                                         fp    = quasar_ivar[i, :],
#    #                                         left  = np.nan,
#    #                                         right = np.nan)

fig = plt.figure(figsize=(18, 6))
for idx in idxs:
    lam_rest = lam_obs / (1 + quasar_z[idx])
    plt.plot(lam_rest, quasar_spectra[idx, :], label="$z = %2.2f$"%quasar_z[idx])
plt.ylim(0, quasar_spectra[idxs, :].max())
plt.legend(fontsize='xx-large')
plt.title("Red-shift comparison of quasar spectra")
plt.xlabel("wavelength")
plt.ylabel("$f(\lambda)$")
plt.savefig(out_dir + "quasar_redshift_rest_frame.pdf", bbox_inches = 'tight')

## plot example spectrograph and overlay SDSS filter bands 
#normalize first spectral density
normalizer = .0012 * integrate.cumtrapz(quasar_spectra[0,:], lam_obs)[-1]
fig = plt.figure(figsize=(18, 6))
plt.ylim(0, (quasar_spectra[0, :]/normalizer).max())
plt.title("Quasar full spectrum", fontsize=16)
plt.xlabel("wavelength", fontsize=16)
plt.ylabel("$f(\lambda)$", fontsize=16)
colors = ['g', 'r', 'c', 'm', 'y', 'k']
for n, b in enumerate(planck.bands): 
    plt.plot(planck.wavelength_lookup[b] * 1e10, 
             planck.sensitivity_lookup[b], 
             color=colors[n], alpha = .5, label = "%s band"%b, linewidth=3)
    plt.fill_between(planck.wavelength_lookup[b] * 1e10, 
                     planck.sensitivity_lookup[b], 
                     alpha=.5, color=colors[n], label="%s band"%b)
plt.plot(lam_obs, quasar_spectra[0, :]/normalizer, label="(scaled) spectrum", linewidth=2)
plt.xlim(3000, 10500)
plt.legend(fontsize='xx-large')
plt.savefig(out_dir + "quasar_spectrum_sdss_filters.pdf", bbox_inches = 'tight')




