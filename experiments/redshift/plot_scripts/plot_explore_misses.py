import fitsio
import numpy as np
import numpy.random as npr
from scipy import interpolate
from scipy.optimize import minimize
from funkyyak import grad, numpy_wrapper as np
from redshift_utils import load_data_clean_split, project_to_bands
from slicesample import slicesample
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sns.set_style("white")
current_palette = sns.color_palette()
npr.seed(42)

## save figure output files
out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"

## load a handful of quasar spectra
lam_obs, qtrain, qtest = \
    load_data_clean_split(spec_fits_file = 'quasar_data.fits', Ntrain = 400)

## load in basis
th  = np.load("cache/basis_th.npy")
lls = np.load("cache/lls.npy")
lam0 = np.load("cache/lam0.npy")
N    = th.shape[1] - lam0.shape[0]
omegas = th[:,:N]
betas  = th[:, N:]
W = np.exp(omegas)
B = np.exp(betas)
B = B / B.sum(axis=1, keepdims=True)

## compute all marginal expected z's and compare 
z_pred = np.zeros(qtest['Z'].shape)
z_lo   = np.zeros(qtest['Z'].shape)
z_hi   = np.zeros(qtest['Z'].shape)
for n in range(qtest['Z'].shape[0]):
    z_n = qtest['Z'][n]
    spec_n = qtest['spectra'][n, :]
    if os.path.exists("cache_remote/cache/ll_samps_train_idx_%d.npy"%n):
        ll_samps = np.load("cache_remote/cache/ll_samps_train_idx_%d.npy"%n)
        th_samps = np.load("cache_remote/cache/th_samps_train_idx_%d.npy"%n)
        Nsamps = th_samps.shape[0]

        ## reconstruct the basis from samples
        samp_idxs = np.arange(Nsamps/2, Nsamps)
        z_pred[n]        = th_samps[samp_idxs, -1].mean()
        z_lo[n], z_hi[n] = np.percentile(th_samps[samp_idxs, -1], [.5, 99.5])
    else: 
        print "missing %d"%n
        z_pred[n] = np.nan
        continue
z_test = qtest['Z']



## figure out the Max Likelihood weight value with respect to each test example
#def loss_omegas(omegas, B, X, inv_var): 
#    ll_omega = 1 / (100.) * np.sum(np.square(omegas))
#    Xtilde   = np.dot(np.exp(omegas), B)
#    return np.sum( inv_var * np.square(X - Xtilde) ) + ll_omega
#loss_omegas_grad = grad(loss_omegas)

What = np.zeros((len(z_test), B.shape[0]))
for n in range(len(z_test)):
    spec_n = qtest['spectra'][n, :]
    ivar_n = qtest['spectra_ivar'][n, :]
    z_n    = qtest['Z'][n]
    What[n, :] = fit_weights_given_basis(B, lam0, spec_n, ivar_n, z_n, lam_obs, sgd_iter=100)

    #convert spec_n to lam0
    #spec_n_resampled = np.interp(lam0, lam_obs/(1+z_n), spec_n, left=np.nan, right=np.nan)
    #ivar_n_resampled = np.interp(lam0, lam_obs/(1+z_n), ivar_n, left=np.nan, right=np.nan)
    #spec_n_resampled[np.isnan(spec_n_resampled)] = 0.0
    #ivar_n_resampled[np.isnan(ivar_n_resampled)] = 0.0
    #omegas = .01*npr.randn(B.shape[0])
    #res = minimize(x0  = omegas,
    #               fun = lambda(th): loss_omegas(th, B, spec_n_resampled, ivar_n_resampled), 
    #               jac = lambda(th): loss_omegas_grad(th, B, spec_n_resampled, ivar_n_resampled),
    #               method = 'L-BFGS-B',
    #               options = { 'disp': False, 'maxiter': 1000 })
    #What[n, :] = np.exp(res.x)

# visualize What in 2-d
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
Wproj = pca.fit_transform(np.log(What))
plt.scatter(Wproj[:,0], Wproj[:, 1])

## figure out the big misses - remove the NAN
dists    = np.abs(z_test - z_pred)
miss_idx = dists.argsort()[::-1][1:]
for idx in miss_idx[1:30]: 
    plt.text(Wproj[idx,0], Wproj[idx,1], s=idx)
plt.show()

## look at the total magnitude of the latent function (sum of W's)
plt.scatter(z_test, z_pred, s = .05*What.sum(axis=1))
for n in range(len(z_test)):
    plt.text(z_test[n], z_pred[n], "test: %d"%n)
plt.show()

## juxtapose the best fits against the worst fits
miss_idx = dists.argsort()[::-1][1:]
fig, axarr = plt.subplots(2, 1)
for idx in miss_idx[0:10]:
    axarr[0].plot(lam_obs/(1+qtest['Z'][idx]), qtest['spectra'][idx, :], 
                  label="z = %2.2f"%qtest['Z'][idx])
axarr[0].legend()
for idx in miss_idx[-1:-10:-1]:
    axarr[1].plot(lam_obs/(1+qtest['Z'][idx]), qtest['spectra'][idx, :], 
                  label="z = %2.2f"%qtest['Z'][idx])
axarr[1].legend()
plt.show()

## plot some reconstructions for the baddies, and the good ones
Nshow = 3
fig, axarr = plt.subplots(Nshow, 1)
for n, idx in enumerate(miss_idx[0:Nshow]):
    axarr[n].plot(lam_obs, qtest['spectra'][idx,:], label="$z = %2.2f"%qtest['Z'][idx])
    axarr[n].plot(lam0 * (1 + qtest['Z'][idx]), What[idx,:].dot(B))
    axarr[n].plot(lam_obs, qtest['spectra_ivar'][idx,:], alpha = .5, color = 'grey')
    axarr[n].set_xlim(lam_obs[0], lam_obs[-1])
    axarr[n].set_ylim(qtest['spectra'][idx,:].min(), qtest['spectra'][idx,:].max())
    axarr[n].set_title("$|z_{spec} - z_{photo}| = %2.2f"%dists[idx])
    axarr[n].legend()

## plot some reconstructions for the baddies, and the good ones
Nshow = 3
fig, axarr = plt.subplots(Nshow, 1)
for n, idx in enumerate(miss_idx[-1:(-Nshow-1):-1]):
    axarr[n].plot(lam_obs, qtest['spectra'][idx,:], label="$z = %2.2f"%qtest['Z'][idx])
    axarr[n].plot(lam0 * (1 + qtest['Z'][idx]), What[idx,:].dot(B))
    axarr[n].plot(lam_obs, qtest['spectra_ivar'][idx,:], alpha = .5, color = 'grey')
    axarr[n].set_xlim(lam_obs[0], lam_obs[-1])
    axarr[n].set_ylim(qtest['spectra'][idx,:].min(), qtest['spectra'][idx,:].max())
    axarr[n].set_title("$|z_{spec} - z_{photo}| = %2.2f"%dists[idx])
    axarr[n].legend()
plt.show()


