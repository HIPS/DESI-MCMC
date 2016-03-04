#import numpy as np
import fitsio
import sys
sys.path.append("../..")
from scipy import interpolate
#from celeste import F0itsImage, celeste_likelihood_multi_image, gen_model_image
#from util.init_utils import load_imgs_and_catalog
import numpy as np
import matplotlib.pyplot as plt
from redshift_utils import load_data_clean_split, project_to_bands
from glob import glob
import scipy.integrate as integrate

## grab some plotting defaults
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
idxs = [11, 23, 4]

# plot multiple and then plot multiple shifted

idxs = [23, 4] # 23] #, 4]
fig, axarr = plt.subplots(2, 1, figsize=(18, 6))
for idx in idxs:
    quasar_spectra[idx, quasar_spectra[idx,:].argmax()] = 0

## plot OBSERVATION frame
for i, idx in enumerate(idxs):
    axarr[0].plot(lam_obs, quasar_spectra[idx, :].T, color = sns.color_palette()[i], 
                  label="$z = %2.2f$"%quasar_z[idx], alpha=.75)
axarr[0].set_ylim(0, 25) #quasar_spectra[idxs, :].max())
axarr[0].set_xlim(2000, lam_obs.max())
axarr[0].legend(fontsize='xx-large')
axarr[0].set_title("Red-shift comparison of quasar spectra")
axarr[0].set_ylabel("$f^{(obs)}(\lambda)$", fontsize=18)
axarr[0].xaxis.set_tick_params(labelsize=15)

## plot rest frame
peak_lam = [4100, 5700]
pts = [ [(.185, .32),
         (.35, .78)],
        [(.185, .46), 
         (.35, .8)]
         ]
for i,idx in enumerate(idxs):
    lam_rest = lam_obs / (1 + quasar_z[idx])
    axarr[1].plot(lam_rest, quasar_spectra[idx, :].T, 
                  color = sns.color_palette()[i], label="$z = %2.2f$"%quasar_z[idx], alpha=.75)
    # draw line between plots
    line = matplotlib.lines.Line2D(pts[i][0], #(coord1[0], coord1[0]),
                                   pts[i][1], #(coord1[1], coord2[1]),
                                   transform = fig.transFigure, color=sns.color_palette()[i], 
                                   linewidth=4)
    fig.lines.append(line)

axarr[1].set_ylim(0, 25) # quasar_spectra[idxs, :].max())
axarr[1].set_xlim(900, 5000) #lam_rest.max()) #lam_obs.max()-2000)
axarr[1].legend(fontsize='xx-large')
axarr[1].set_xlabel("wavelength $(\AA)$", fontsize=18)
axarr[1].set_ylabel("$f^{(rest)}(\lambda)$", fontsize=18)
axarr[1].xaxis.set_tick_params(labelsize=15)
plt.savefig(out_dir + "quasar_redshift_example.pdf", bbox_inches = 'tight')


