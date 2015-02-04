import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
from redshift_utils import load_data_clean_split, project_to_bands
from quasar_fit_basis import load_basis_fit
from quasar_sample_basis import load_basis_samples
from glob import glob
npr.seed(42)

## grab some plotting defaults
sns.set_style("ticks")
current_palette = sns.color_palette()

if __name__=="__main__":

    ### Paper out directory
    out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"

    ### load MCMC sample files
    sample_files = glob('cache_remote/photo_experiment0/basis_samples_K-4_V-1364_chain_*.npy')
    chains = {}
    for sfile in sample_files:
        th_samples, ll_samps, lam0, lam0_delta, parser, chain_idx = \
            load_basis_samples(sfile)
        chains[chain_idx] = {'th':th_samples, 'lls':ll_samps, 'parser':parser}

    # visualize LL trace for each chain
    for i in range(len(chains)): 
        plt.plot(chains[i]['lls'][-1000:])
    plt.show()

    ### load MLE basis 
    th, lam0, lam0_delta, parser = load_basis_fit('cache/basis_fit_K-4_V-1364.pkl')
    mus    = parser.get(th, 'mus')
    betas  = parser.get(th, 'betas')
    omegas = parser.get(th, 'omegas')
    W_mle  = np.exp(omegas)
    W_mle /= np.sum(W_mle, axis=1, keepdims=True)
    B_mle  = np.exp(betas)
    B_mle /= np.sum(B_mle * lam0_delta, axis=1, keepdims=True)
    M_mle = np.exp(mus)

    ### load training data
    lam_obs, qtrain, qtest = \
        load_data_clean_split(spec_fits_file = 'quasar_data.fits',
                              Ntrain = 400)
    N = qtrain['spectra'].shape[0]

    ##################################################################
    # inspect posterior summary of betas
    ##################################################################
    # Unpack Samples
    chain_idx = 2
    th_samps = chains[2]['th'][-3000:, :]
    Nsamps   = th_samps.shape[0]
    K, V     = chains[2]['parser'].idxs_and_shapes['betas'][1]
    Ntrain   = chains[2]['parser'].idxs_and_shapes['mus'][1][0]

    # set up prior cholesky
    sig2_omega = 1.
    sig2_mu    = 500.
    beta_kern = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=40) #length_scale)
    K_beta    = beta_kern.K(lam0.reshape((-1, 1)))
    K_chol    = np.linalg.cholesky(K_beta)
    K_inv     = np.linalg.inv(K_beta)

    B_samps = np.zeros((Nsamps, K, V))
    W_samps = np.zeros((Nsamps, Ntrain, K))
    M_samps = np.zeros((Nsamps, Ntrain))
    for n in range(Nsamps):
        betas = K_chol.dot(parser.get(th_samps[n, :], 'betas').T).T
        #betas = parser.get(th_samps[n, :], 'betas')
        mus   = parser.get(th_samps[n, :], 'mus')
        omegas = parser.get(th_samps[n, :], 'omegas')
        W = np.exp(omegas)
        W /= np.sum(W, axis=1, keepdims=True)
        B = np.exp(betas)
        B /= np.sum(B * lam0_delta, axis=1, keepdims=True)
        M = np.exp(mus)
        B_samps[n, :] = B
        M_samps[n, :] = M.squeeze()
        W_samps[n, :] = W


    ######################################################################
    ## Plot Basis
    ######################################################################
    fig, axarr = plt.subplots(K, 1, figsize=(16, 10))
    ranges = [[1000, 4000], [800, 1800], [0, 10000], [800, 2000]]
    for k in range(K):
        #plt.plot(lam0, B_samps.mean(axis=0).T)
        axarr[k].plot(lam0, B_samps[2000, k, :], color=sns.color_palette()[k])
        axarr[k].set_xlim(ranges[k])
        #if k != 3:
        #    axarr[k].set_xticks([])
    plt.xlabel(" wavelength $(\AA)$ ", fontsize=16, labelpad=20)
    sns.despine(top=True)
    plt.savefig(out_dir + "basis_samp_K_%d.pdf"%K, bbox_inches="tight")
    plt.close('all')

    #######################################################################
    ### Get a handle on basis marginasl
    #######################################################################
    qidx = 30
    n, bins, patches = plt.hist(M_samps[:, qidx], 25, normed=True)
    plt.vlines(M_mle[qidx], ymin=0, ymax=n.max())
    plt.show()

    n, bins, patches = plt.hist(W_samps[:, qidx, 1], 30, normed=True)
    plt.vlines(W_mle[qidx, 1], ymin=0, ymax=n.max(), linewidth=3 )

    def plot_recon(spec_recon, z, spec, spec_ivar):
        fig = plt.figure(figsize=(18,6))
        plt.plot(lam_obs / (1 + z), spec, linewidth=2,
                 color=current_palette[0], label="measured")
        plt.plot(lam_obs / (1 + z), spec_ivar, 
                 linewidth=1, color='grey', alpha=.5, label="inverse variance")
        plt.plot(lam0, spec_recon, linewidth=2, color=current_palette[2], label="rank %d reconstruction"%B.shape[0])
        plt.xlim(lam_obs.min()/(1 + z), lam_obs.max()/(1+z))
        plt.ylim(0, spec.max())
        plt.xlabel("wavelength")
        plt.ylabel("spectrum")
        plt.legend(fontsize='xx-large')

    qidx = 30
    samp_idx = 2999
    spec_recon = M_samps[samp_idx, qidx] * \
                 W_samps[samp_idx, qidx, :].dot(B_samps[samp_idx,:,:])
    #spec_recon = np.dot(M_mle[qidx] * W_mle[qidx, :], B_mle)
    plot_recon(spec_recon, qtrain['Z'][qidx], 
                           qtrain['spectra'][qidx,:],
                           qtrain['spectra_ivar'][qidx,:])

    plt.show()


