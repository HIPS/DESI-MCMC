import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
from redshift_utils import load_data_clean_split, project_to_bands
npr.seed(42)

## grab some plotting defaults
sns.set_style("white")
current_palette = sns.color_palette()

#######################################################################
out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"

## plot the basis
th  = np.load("cache/basis_th.npy")
lls = np.load("cache/lls.npy")
lam0 = np.load("cache/lam0.npy")
N    = th.shape[1] - lam0.shape[0]
omegas = th[:,:N]
betas  = th[:, N:]
W = np.exp(omegas)
B = np.exp(betas)
B = B / B.sum(axis=1, keepdims=True)
fig = plt.figure(figsize=(18,6))
plt.plot(lam0, B.T, linewidth=2)
plt.title("Rest frame basis")
plt.xlabel("wavelength")
plt.ylabel("normalized spectrum")
plt.ylim(0, .02)
plt.xlim(500, 10000)
plt.savefig(out_dir + "rank_%d_basis.pdf"%W.shape[0], bbox_inches='tight')

## plot a reconstruction
lam_obs, qtrain, qtest = \
    load_data_clean_split(spec_fits_file = 'quasar_data.fits', 
                          Ntrain = 400)
quasar_spectra = qtrain['spectra']
quasar_z       = qtrain['Z']
quasar_ivar    = qtrain['spectra_ivar']
quasar_zerr    = qtrain['Z_err']
N              = quasar_spectra.shape[0]
idxs = [0, 1, 4, 8]
for idx in idxs:
    spec_recon = W[:, idx].dot(B)
    fig = plt.figure(figsize=(18,6))
    plt.plot(lam_obs / (1 + quasar_z[idx]), 
             quasar_spectra[idx,:], linewidth=2,  color=current_palette[0], label="measured")
    plt.plot(lam_obs / (1 + quasar_z[idx]), 
             quasar_ivar[idx, :], linewidth=1, color='grey', alpha=.5, label="inverse variance")
    plt.plot(lam0, spec_recon, linewidth=2, color=current_palette[2], label="rank %d reconstruction"%W.shape[0])
    plt.xlim(lam_obs.min()/(1 + quasar_z[idx]), lam_obs.max()/(1+quasar_z[idx]))
    plt.ylim(0, quasar_spectra[idx,:].max())
    plt.xlabel("wavelength")
    plt.ylabel("spectrum")
    plt.legend()
    plt.savefig(out_dir + "idx_%d_rank_%d_reconstruction.pdf"%(idx, W.shape[0]), bbox_inches='tight')


## visualize the distribution of the weights
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
Wproj = pca.fit_transform(np.log(W.T))
fig = plt.figure()
plt.scatter(Wproj[:, 0], Wproj[:, 1])
plt.savefig(out_dir + "weight_pca_rank_%d.pdf"%W.shape[0], bbox_inches='tight')

