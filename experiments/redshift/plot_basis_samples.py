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
sns.set_style("white")
current_palette = sns.color_palette()


if __name__=="__main__":

    sample_files = glob('cache/basis_samples_K-4_V-1364_chain_*.npy')
    chains = {}
    for sfile in sample_files:
        th_samples, ll_samps, lam0, lam0_delta, parser, chain_idx = \
            load_basis_samples(sfile)
        chains[chain_idx] = {'th':th_samples, 'lls':ll_samps, 'parser':parser}

    # visualize LL trace for each chain
    for ch in chains.itervalues(): 
        plt.plot(ch['lls'])
    plt.show()

    ##################################################################
    # inspect posterior summary of betas
    ##################################################################
    # Unpack Samples
    B_samps = np.zeros((Nsamps, K, V))
    W_samps = np.zeros((Nsamps, Ntrain, K))
    M_samps = np.zeros((Nsamps, Ntrain))
    for n in range(Nsamps):
        betas = K_chol.dot(parser.get(th_samps[n, :], 'betas').T).T
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

    plt.plot(lam0, B_samps.mean(axis=0).T)
    plt.show()

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

    qidx = 20
    samp_idx = 1409
    spec_recon = M_samps[samp_idx][qidx] * \
                 W_samps[samp_idx, qidx, :].dot(B_samps[samp_idx,:,:])
    #spec_recon = np.dot(M_mle[qidx] * W_mle[qidx, :], B_mle)
    plot_recon(spec_recon, qtrain['Z'][qidx], 
                           qtrain['spectra'][qidx,:],
                           qtrain['spectra_ivar'][qidx,:])

    plt.show()


