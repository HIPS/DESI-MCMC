import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
from redshift_utils import load_data_clean_split, project_to_bands, get_lam0
from quasar_fit_basis import load_basis_fit
from quasar_sample_basis import load_basis_samples
npr.seed(42)

## grab some plotting defaults
sns.set_style("white")
current_palette = sns.color_palette()

def make_params(th, parser, lam0_delta):
    mus    = parser.get(th, 'mus')
    betas  = parser.get(th, 'betas')
    omegas = parser.get(th, 'omegas')
    W = np.exp(omegas)
    W = W / np.sum(W, axis=1, keepdims=True)
    B = np.exp(betas)
    B = B / np.sum(B * lam0_delta, axis=1, keepdims=True)
    M = np.exp(mus)
    return W, B, M

#######################################################################
out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"
V = 1364 #2728        #  "1364"  or 2728
bdir = "/Users/acm/Proj/DESIMCMC/experiments/redshift/cache/basis_locked/"
bname = "basis_fit_K-4_V-1364_split-random.pkl"
th, lam0, lam0_delta, parser = load_basis_fit(os.path.join(bdir, bname))
W_mle, B_mle, M_mle = make_params(th, parser, lam0_delta)

# basis for figure
#bfile = "/UAsers/acm/Dropbox/Proj/astro/DESIMCMC/experiments/redshift/cache/old/basis_th_K-4_V-%d.npy"%V
#f = open(bfile, 'rb')
#lam0, lam0_delta = get_lam0(lam_subsample=5)
#a = np.load(bfile)
#W_mle = np.exp(a[:, :400])
#W_mle = W_mle / np.sum(W_mle, axis=1, keepdims=True)
#B_mle = np.exp(a[:, 400:])
#B_mle = B_mle / np.sum(B_mle * lam0_delta, axis=1, keepdims=True)
#K = B_mle.shape[0]

######################################################################
## PLOT MLE BASIS
######################################################################
fig, axarr = plt.subplots(K, 1, figsize=(12, 6))
ranges = [[800, 8000], [100, 8000], [800, 4000], [800, 2000]]
for k in range(K):
    axarr[k].plot(lam0, B_mle[k, :], color=sns.color_palette()[k])
    axarr[k].set_xlim(ranges[k])
    axarr[k].tick_params(axis='both', which='major', labelsize=12)
    axarr[k].tick_params(axis='x', which='major', labelsize=16)
    max_yticks = 3
    yloc = plt.MaxNLocator(max_yticks)
    axarr[k].yaxis.set_major_locator(yloc)
plt.xlabel(" $\lambda$ (wavelength in $\AA$) ", fontsize=20, labelpad=20)
plt.text(.02, .5, "$B_k(\lambda)$", transform = fig.transFigure, fontsize=20)
plt.subplots_adjust(hspace=.6)
#plt.tight_layout()
sns.despine(top=True)
plt.savefig("/Users/acm/Proj/DESIMCMC/tex/quasar_z/figs/rank_%d_basis-random.pdf"%B.shape[0], 
            bbox_inches='tight')
plt.close('all')

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
    spec_recon = M[idx]*W[idx,:].dot(B)
    fig = plt.figure(figsize=(18,3))
    plt.plot(lam_obs / (1 + quasar_z[idx]), 
             quasar_spectra[idx,:], linewidth=2,  color=current_palette[0], label="measured")
    plt.plot(lam_obs / (1 + quasar_z[idx]), 
             quasar_ivar[idx, :], linewidth=1, color='grey', alpha=.25, label="inverse variance")
    plt.plot(lam0, spec_recon, linewidth=2, color=current_palette[2], label="rank %d reconstruction"%B.shape[0])

    # set limits
    plt.xlim(lam_obs.min()/(1 + quasar_z[idx]), lam_obs.max()/(1+quasar_z[idx]))
    # hack for idx 50
    if idx == 0:
        plt.ylim(0, 45)
    else:
        plt.ylim(0, quasar_spectra[idx,:].max())

    plt.xlabel("wavelength ($\AA$)", fontsize=16)
    plt.ylabel("SED", fontsize=16, rotation='horizontal', labelpad=20)
    plt.legend(fontsize=16)
    plt.savefig(out_dir + "idx_%d_rank_%d_reconstruction.pdf"%(idx, B.shape[0]), bbox_inches='tight')
    plt.close('all')

## visualize the distribution of the weights
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
Wproj = pca.fit_transform(np.log(W))
fig = plt.figure()
plt.scatter(Wproj[:, 0], Wproj[:, 1])
plt.savefig(out_dir + "weight_pca_rank_%d.pdf"%W.shape[1], bbox_inches='tight')
plt.close('all')

## print the two distributions of z scores
fig, axarr = plt.subplots(2, 1, figsize=(10, 5))
n, bins, patches = axarr[0].hist(qtrain['Z'], 20)
axarr[1].hist(qtest['Z'], bins)
plt.savefig(out_dir + "train_test_z_scores.pdf")
plt.close('all')

