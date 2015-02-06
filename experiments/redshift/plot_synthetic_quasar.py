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
idxs = [4, 11, 23]

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
fig = plt.figure(figsize=(18, 4))
quasar_spectra[idxs[0], quasar_spectra[idxs[0],:].argmax()] = 0
for idx in idxs:
    plt.plot(lam_obs, quasar_spectra[idx, :].T, label="$z = %2.2f$"%quasar_z[idx], alpha=.75)
plt.ylim(0, quasar_spectra[idxs, :].max())
plt.legend(fontsize='xx-large')
plt.title("Red-shift comparison of quasar spectra")
#plt.xlabel("wavelength")
plt.ylabel("$f^{(obs)}(\lambda)$", fontsize=18)
plt.savefig(out_dir + "quasar_redshift_obs_frame.pdf", bbox_inches = 'tight')

fig = plt.figure(figsize=(18, 4))
for idx in idxs:
    lam_rest = lam_obs / (1 + quasar_z[idx])
    plt.plot(lam_rest, quasar_spectra[idx, :], label="$z = %2.2f$"%quasar_z[idx], alpha=.75)
plt.ylim(0, quasar_spectra[idxs, :].max())
plt.legend(fontsize='xx-large')
#plt.title("Red-shift comparison of quasar spectra")
plt.xlabel("wavelength $(\AA)$", fontsize=18)
plt.ylabel("$f^{(rest)}(\lambda)$", fontsize=18)
plt.savefig(out_dir + "quasar_redshift_rest_frame.pdf", bbox_inches = 'tight')

## plot a bunch in it's own directory
full_out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/quasar_specs/" 
full_idx = np.arange(0, qtrain['spectra'].shape[0], 5)
for idx in full_idx: 
    fig = plt.figure(figsize=(20,6))
    lam_rest = lam_obs / (1 + qtrain['Z'][idx])
    plt.plot(lam_rest, qtrain['spectra'][idx, :].T, label="z = %2.2f"%qtrain['Z'][idx], linewidth=1)
    plt.plot(lam_rest, qtrain['spectra_ivar'][idx, :].T, label="inv var", color='grey', alpha=.5, linewidth=.3)
    plt.title("Quasar (training # %d)"%idx)
    plt.xlabel("wavelength")
    plt.ylabel("$f(\lambda)$", fontsize=18)
    plt.xlim(500, 4500)
    plt.legend()
    plt.savefig(full_out_dir + "quasar_spectra_%d.png"%idx, bbox_inches = 'tight')
    #plt.close('all')

## plot example spectrograph and overlay SDSS filter bands 
#normalize first spectral density

## plot double
spec_file = glob("../../data/DR10QSO/specs/spec-*-*-*.fits")[5]
sdf = fitsio.FITS(spec_file)
spec_flux = sdf[1]['flux'].read()
spec_lam  = np.power(10., sdf[1]['loglam'].read())
psf_flux = sdf[2]['PSFFLUX'].read()
psf_flux_ivar = sdf[2]['PSFFLUX_IVAR'].read()
spec_flux[spec_flux.argmax()] = 0

normalizer = .0072 * integrate.cumtrapz(spec_flux, spec_lam)[-1]
fig = plt.figure(figsize=(18, 4))
plt.ylim(0, (quasar_spectra[0, :]/normalizer).max())
#plt.title("Quasar full spectrum", fontsize=16)
plt.xlabel("wavelength $(\AA)$", fontsize=20)
plt.ylabel("$f^{(obs)}(\lambda)$", fontsize=20)
colors = ['g', 'r', 'c', 'm', 'y', 'k']
plt.plot(spec_lam, 2.5*spec_flux/normalizer, label="SED", linewidth=2, alpha=.95)
for n, b in enumerate(planck.bands): 
    plt.plot(planck.wavelength_lookup[b] * 1e10, 
             planck.sensitivity_lookup[b], 
             color=colors[n], alpha = .5, label = "%s band"%b, linewidth=3)
    plt.fill_between(planck.wavelength_lookup[b] * 1e10, 
                     planck.sensitivity_lookup[b], 
                     alpha=.5, color=colors[n], label="%s band"%b)
plt.xlim(3000, 10500)
plt.ylim(0, .55)
plt.xticks(fontsize=16)
plt.legend(fontsize='xx-large', ncol=2)
plt.savefig(out_dir + "quasar_spectrum_sdss_filters.pdf", bbox_inches = 'tight')
plt.close('all')

## plot corresponding fluxes and uncertainties 
fig = plt.figure(figsize=(6, 4))
xs = np.arange(5)
plt.bar(xs, psf_flux, alpha=.4, width=.8, yerr = 2*np.sqrt(1./spec_ivar),
        error_kw = {'linewidth':5},
        color=sns.color_palette()[1], label='PSFFLUX')
plt.legend(loc='upper left', fontsize=18)
plt.ylabel('flux (nanomaggies)', fontsize=18, labelpad=10)
plt.xlabel("band", fontsize = 18)
plt.xticks(xs + .4, ['u', 'g', 'r', 'i', 'z'], fontsize=18)
plt.savefig(out_dir + "quasar_spectrum_bands.pdf", bbox_inches='tight')


### define a set of quasars to look at, and grab their pixel values
#for i, n in enumerate([0]): 
#    mu_n        = project_to_bands(np.atleast_2d(quasar_spectra[n, :]), lam_obs).ravel()
#    #x_n         = npr.poisson(mu_n).ravel()
#
#    ## Generate some fake sources using real image data (PSF and stuff)
#    cat_glob = glob('../../data/stamp_catalog/cat*.fits')[0:1]
#    srcs, imgs, teff_catalog, us = load_imgs_and_catalog(cat_glob)
#
#    ## create quasar source with known mean values for each band
#    np.random.seed(42)
#    srcs = srcs[1:]
#    srcs[0].t = None
#    srcs[0].b = None
#    srcs[0].fluxes = dict(zip(['u', 'g', 'r', 'i', 'z'], mu_n))
#
#    # re-generate images using these source params
#    for img in imgs: 
#        mimg      = gen_model_image(srcs, img)
#        img.nelec = np.random.poisson(mimg)
#        plt.imshow(img.nelec.T, interpolation='none')
#        plt.savefig(out_dir + "img_%d_band_%s.pdf"%(n, img.band), bbox_inches='tight')
#
#
### plot the raw fluxes for each band
#mus = project_to_bands(quasar_spectra[idxs, :], lam_obs.ravel())
#mus /= mus.sum(axis=1, keepdims=True)
#fig = plt.figure()
#plt.imshow(mus, interpolation='nearest')
#plt.xlabel('image band')
#plt.ylabel('quasar index')
#plt.title('Quasar flux distributions' )
#plt.xticks(np.arange(mus.shape[1]), ['u', 'g', 'r', 'i', 'z'], fontsize=10)
#plt.yticks(np.arange(mus.shape[0]), idxs)
#plt.colorbar()
#plt.savefig(out_dir + "quasar_sdss_fluxes.pdf", bbox_inches = 'tight', dpi=400)


