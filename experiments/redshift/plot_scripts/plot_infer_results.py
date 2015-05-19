import fitsio
import numpy as np
import numpy.random as npr
from scipy.optimize import minimize
from scipy import interpolate
from redshift_utils import load_data_clean_split, project_to_bands
from slicesample import slicesample
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
from glob import glob
#from quasar_infer_photometry import pixel_likelihood, prior_w, prior_z
from quasar_fit_basis import load_basis_fit
from quasar_infer_photometry import load_redshift_samples
from scrape_quasar_spectra import download_spec_file
from scipy.stats import kde
sns.set_style("white")
current_palette = sns.color_palette()
npr.seed(42)

##############################################################################
## save figure output files
##############################################################################
out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"
#experiment_id = "photo_experiment0"

##########################################################################
## Load in spectroscopically measured quasars + fluxes
##########################################################################
#qso_df  = fitsio.FITS('../../data/DR10QSO/DR10Q_v2.fits')[1].read()
#
# set TEST INDICES 
#npr.seed(13)
#test_idx = npr.permutation(len(qso_df))

##########################################################################
## Load in basis fit (or basis samples)
##########################################################################
### load ML basis from cache (beta and omega values)
basis_cache = 'cache/basis_fit_K-4_V-2728.pkl'
USE_CACHE = True
if os.path.exists(basis_cache) and USE_CACHE:
    th, lam0, lam0_delta, parser = load_basis_fit(basis_cache)
# compute actual weights and basis values (normalized basis + weights)
mus    = parser.get(th, 'mus')
betas  = parser.get(th, 'betas')
omegas = parser.get(th, 'omegas')
W = np.exp(omegas)
W = W / np.sum(W, axis=1, keepdims=True)
B = np.exp(betas)
B = B / np.sum(B * lam0_delta, axis=1, keepdims=True)
M = np.exp(mus)

##############################################################################
## saved quasar-specific files
##############################################################################
qso_sample_files = glob('cache_remote/temper_experiment1/redshift_samples*.npy')
qso_sample_files.sort()

##############################################################################
## Z_spec vs Z_phot plot 
##############################################################################
if False:
    z_pred = np.zeros(len(qso_sample_files))
    z_pred_mode = np.zeros(len(qso_sample_files))
    z_lo   = np.zeros(len(z_pred))
    z_hi   = np.zeros(len(z_pred))
    z_true = np.zeros(len(z_pred))
    z_lo0 = np.zeros(len(z_pred))
    z_hi0 = np.zeros(len(z_pred))
    q_inds = np.zeros(len(z_pred))
    mode_sample = np.zeros(len(z_pred), dtype=np.int)
    expected_m = np.zeros(len(z_pred))
    expected_w = np.zeros((len(z_pred), 4))
    for i, qso_samp_file in enumerate(qso_sample_files):
        if i%25==0: print "%d of %d"%(i, len(qso_sample_files))
        try: 
            th_samps, lls, q_idx, qso_info, chain_idx = load_redshift_samples(qso_samp_file)
        except:
            print "skipping %d"%i
            continue
        q_inds[i] = q_idx

        Nsamps = th_samps.shape[0]
        # compare predict to true
        z_true[i]        = qso_info['Z_VI']
        z_pred[i]        = th_samps[Nsamps/2:, 0].mean()
        z_lo[i], z_hi[i] = np.percentile(th_samps[Nsamps/2:, 0], [.5, 99.5])
        z_lo0[i], z_hi0[i] = np.percentile(th_samps[Nsamps/2:, 0], [5, 95])

        # kernel density estimate to find the highest mode
        z_unique = np.unique(th_samps[Nsamps/2:, 0])
        density  = kde.gaussian_kde(z_unique, bw_method = .08 ) #'silverman')
        mode_sample[i] = density(z_unique).argmax()
        z_pred_mode[i] = z_unique[ mode_sample[i] ]

        # expected magnitude/
        expected_m[i] = np.exp(th_samps[Nsamps/2:, -1]).mean()

        # expected weights
        ws = np.exp(th_samps[Nsamps/2:, 1:-1])
        ws /= np.sum(ws, axis=1, keepdims=True)
        expected_w[i, :] = ws.mean(axis=0)

fig = plt.figure(figsize=(8, 8))
max_z = max(z_true.max(), z_pred_mode[np.isfinite(z_pred)].max()) + .2
min_z = -.2
plt.plot([min_z, max_z], [min_z, max_z], linewidth=2, alpha=.5)
plt_idx = np.arange(0, len(z_pred), 2)
for n in plt_idx[::2]: 
    plt.plot([z_true[n], z_true[n]], [z_lo0[n], z_hi0[n]], alpha = .15, color = 'grey', linewidth=1)
plt.scatter(z_true[plt_idx], z_pred[plt_idx], color=current_palette[2], alpha = 1.0)
plt.xlim(min_z, max_z)
plt.ylim(min_z, max_z)
plt.xlabel("$z_{spec}$", fontsize=20, labelpad=20)
plt.ylabel("$z_{photo}$", fontsize=20, rotation='horizontal', labelpad=20)
#plt.title("Posterior expectation model predictions", fontsize=14)
plt.savefig(out_dir + "red-shift-test-predictions.pdf", bbox_inches='tight')

## compute statistics of the sample
lbs = [0, 1, 2, 3, 4, 4.5, 5.]
maes  = np.zeros(len(lbs))
mapes = np.zeros(len(lbs))
maes_mode  = np.zeros(len(lbs))
mapes_mode = np.zeros(len(lbs))
num_ex = np.zeros(len(lbs))
num_cover = np.zeros(len(lbs))
num_cover0 = np.zeros(len(lbs))
for i, lo in enumerate(lbs):
    idx_greater = z_true > lo
    num_ex[i] = np.sum(idx_greater)
    maes[i]  = np.mean(np.abs(z_pred[idx_greater] - z_true[idx_greater]))
    mapes[i] = np.mean(np.abs(z_pred[idx_greater] - z_true[idx_greater])/np.abs(z_true[idx_greater]))
    num_cover[i] = np.sum( (z_true[idx_greater] > z_lo[idx_greater]) & \
                           (z_true[idx_greater] < z_hi[idx_greater]) ) / float(num_ex[i])
    num_cover0[i] = np.sum((z_true[idx_greater] > z_lo0[idx_greater]) & \
                           (z_true[idx_greater] < z_hi0[idx_greater]) ) / float(num_ex[i])
    maes_mode[i] = np.mean(np.abs(z_pred_mode[idx_greater] - z_true[idx_greater]))
    mapes_mode[i] = np.mean(np.abs(z_pred_mode[idx_greater] - z_true[idx_greater]) / np.abs(z_true[idx_greater]))


## write error table
tab_file = file(out_dir + "error_table.tex", 'w')
tab_file.write('\\begin{tabular*}{0.95\\textwidth}{%s} \n'%("c"*(len(lbs)+1)))
tab_file.writelines(["\hline", "\\abovespace\\belowspace \n"])
tab_file.write("$z_{spec}$ & " + " & ".join(["$ > %2.1f (%d)$"%(d, num_ex[i]) for i,d in enumerate(lbs)]))
tab_file.write("\\\\ \n")
tab_file.writelines(["\hline \n", "\\abovespace \n"])
tab_file.writelines([
  "mean MAE & " + " & ".join(["%2.2f"%m for m in maes]) + " \\\\ \n", 
  "mode MAE & " + " & ".join(["%2.2f"%m for m in maes_mode]) + " \\\\ \n",
  "mean MAPE & " + " & ".join(["%2.2f"%m for m in mapes]) + " \\\\ \n",
  "mode MAPE & " + " & ".join(["%2.2f"%m for m in mapes_mode]) + " \\\\ \n"
  ])
tab_file.write("\hline \n")
tab_file.writelines([
    "\% in [.5, 99.5] & ", " & ".join(["%2.1f"%(m*100) for m in num_cover]) + " \\\\ \n",
    "\% in [5, 95] & ", " & ".join(["%2.1f"%(m*100) for m in num_cover0]) + "\\\\ \n "
  ])
tab_file.writelines(['\hline', '\end{tabular*}'])
tab_file.close()


## pca on the posterior expected weights
log_w = np.log(expected_w)


#############################################################################
## Individual marginals and pair-wise marginals
#############################################################################
#z_pred = np.zeros(len(qso_sample_files))
#z_lo   = np.zeros(len(z_pred))
#z_hi   = np.zeros(len(z_pred))
#z_true = np.zeros(len(z_pred))
#for i, qso_samp_file in enumerate(qso_sample_files):
#    # load samples
#    th_samps, ll_samps, q_idx, qso_info = load_redshift_samples(qso_samp_file)
#    Nsamps = th_samps.shape[0]
#    # compare predict to true
#    z_true[i]        = qso_info['Z_VI']
#    z_pred[i]        = th_samps[Nsamps/2:, 0].mean()
#    z_lo[i], z_hi[i] = np.percentile(th_samps[Nsamps/2:, 0], [.5, 99.5])

##
## compute all expected Photo Z's with Spec Z's 
##
## figure out a diverse set
dists = np.abs(z_pred - z_true)
order = dists.argsort() #[::-1]


high_misses = np.where((dists > 1.2) & (z_true < .9)) 
low_misses  = np.where((dists > 1.4) & (z_true > 2.1) & (z_pred_mode < 1.))
close_lo    = np.where((dists < .1) & (z_true < 1.4))[0]
close_mid   = np.where((dists < .1) & (z_true > 1.4) & (z_true < 3.5))[0][0:-1:10]
close_hi    = np.where((dists < .1) & (z_true > 4.0))[0][0:-1:10]
to_inspect = np.concatenate([ order[0:50:5], 
                              order[20:-1:45],
                              np.where(z_true > 4)[0],
                              low_misses[0] ])
to_inspect = low_misses[0]
to_inspect = np.concatenate([np.array([490, 214]), to_inspect])
for i in range(len(z_true)):
    if z_true[i] < 1. and z_pred[i] > 2.:
        print "%d: delta Z = %2.3f (z_pred = %2.3f,  z_true = %2.3f)"%(q_inds[i], dists[i], z_pred[i], z_true[i])


save_folder = "close_mid"
to_inspect = close_mid

save_folder = "close_hi"
to_inspect = np.where((dists < .1) & (z_true > 4.0))[0][0:-1:10]

#########################################################################
### Plot statistics of individual samples of note (big/low distance)
#########################################################################
for n in to_inspect:
    ### load red-shift samples
    #th_chains = []
    #ll_chains = []
    #for chain_idx in range(5):
    #    chain_file = "%s%d.npy"%(os.path.splitext(qso_sample_files[n])[0][:-1], chain_idx)
    #    ths, lls, q_idx, qso_info, chain_idx = load_redshift_samples(chain_file)
    #    th_chains.append(ths)
    #    ll_chains.append(lls)
    #ll_means = np.array([ll[(len(ll)/2):].mean() for ll in ll_chains])
    ## choose best performing chain
    #Nsamps_within = th_chains[0].shape[0]
    #th_samps = th_chains[ll_means.argmax()]
    ##th_samps = np.row_stack([ ths[Nsamps_within/2:, :] for ths in th_chains])
    th_samps, lls, q_idx, qso_info, chain_idx = load_redshift_samples(qso_sample_files[n])
    Nsamps = th_samps.shape[0]
    sub_idx = np.arange(Nsamps/2, Nsamps)
    if n == 490:
        sub_idx = np.arange(1500+1110, Nsamps)
    print "    num samps: %d"%Nsamps
    # Unpack samps
    z_samps = th_samps[sub_idx, 0]
    w_samps = np.exp(th_samps[sub_idx, 1:-1])
    w_samps /= np.sum(w_samps, axis=1, keepdims=True)
    m_samps = np.exp(th_samps[sub_idx, -1])
    z_n = qso_info['Z_VI']
    err = np.abs(z_samps.mean() - z_n)
    print "inspecting quasar test case %d (z_spec = %2.2f)"%(n, z_n)
    print "    ... abs error: ", err
 
    ####################################
    ### redshift posterior marginal
    ####################################
    fig = plt.figure(figsize=(8, 5))
    cnts, bins, patches = plt.hist(z_samps, 40, alpha=.35, normed=True)
    z_samplo, z_samphi = np.percentile(z_samps, [.05, 100])
    plt.xlim(z_samplo, z_samphi)

    # kernel density estimate to find the highest mode
    density = kde.gaussian_kde(z_samps, bw_method = .08 ) #'silverman')
    zgrid   = np.linspace(bins.min(), bins.max(), 400)
    plt.plot(zgrid, density(zgrid), color=sns.color_palette()[3])

    # figure out "best" sample
    z_unique = np.unique(z_samps)
    z_maximizing = z_unique[density(z_unique).argmax()]
    arg_max_samp = np.where(z_samps==z_maximizing)[0][0]
    nearest_z_samp = np.abs(z_samps - z_n).argmin()

    bounds = np.where(z_n > bins[:-1])
    if len(bounds[0]) == 0:
        true_height = cnts.max()
    else:
        true_height = max(cnts[bounds[0][-1]], cnts.max() * .5)
    plt.vlines(z_n, 0,  true_height, linewidth=3, color="black", label="$z_{spec}$")
    plt.vlines(z_samps.mean(), 0, cnts[np.where(z_samps.mean() > bins)[0][-1]], linewidth=3, color=sns.color_palette()[2], label="$E[z_{photo} | x]$")
    plt.xlabel("$z_{photo}$", fontsize=24, labelpad=20)
    plt.ylabel("$p(z | \mathbf{y}, B)$", fontsize=24)
    plt.legend(loc='best', fontsize=24)
    plt.title("qso %s: ($|z_{spec}-z_{photo}|$ = %2.2f)"%(qso_info['SDSS_NAME'], err), fontsize=14)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.savefig(out_dir + "quasar_plots/%s/quasar_%d_posterior_z.pdf"%(save_folder, n), bbox_inches='tight')

    plt.close('all')

    ### show posterior correlation between w and z
    fig = plt.figure(figsize=(6,6))
    plt.imshow(np.corrcoef(th_samps.T), interpolation='nearest')
    plt.colorbar()
    tick_labels = np.concatenate([ ["$w_%d$"%d for d in range(th_samps.shape[1]-1)], 
                                    ["$z$"] ])
    plt.xticks(np.arange(th_samps.shape[1]), tick_labels, fontsize='x-large')
    plt.yticks(np.arange(th_samps.shape[1]), tick_labels, fontsize='x-large')
    plt.savefig(out_dir + "quasar_plots/%s/quasar_%d_posterior_cor.pdf"%(save_folder, n), bbox_inches='tight')

    ### yikessss
    plt.close('all')

    #### plot marginal weight distributions
    #fig, axarr = plt.subplots(1, B.shape[0], figsize=(18,6))
    #for i, ax in enumerate(axarr):
    #    cnts, bins, patches = ax.hist(th_samps[(Nsamps/2):, i], 20, alpha=.5, normed=True)
    #    ax.set_xlabel('$w_%d$'%i)
    #    ax.set_ylabel('$p(w_%d | x)$'%i)
    ##plt.tight_layout()
    #plt.subplots_adjust(top=0.92)
    #plt.suptitle("Basis Loading Marginals")
    #plt.savefig(out_dir + "quasar_plots/quasar_%d_posterior_weights.pdf"%n, bbox_inches='tight')


    ############################################################
    ### Plot expected spectral reconstruction 
    ############################################################
    #### Load in spec measurement corresponding to this example
    spec_file = download_spec_file(qso_info['PLATE'], qso_info['MJD'], qso_info['FIBERID'], redownload=True)
    spec_df  = fitsio.FITS(spec_file)[1].read()
    lam_obs  = np.power(10., spec_df['loglam'])
    spec_obs = spec_df['flux']
    spec_ivar = spec_df['ivar']
    spec_mod = spec_df['model']

    # compute all w's and ms
    f_samps = w_samps.dot(B) 
    recon_samps_resampled = np.zeros((f_samps.shape[0], len(lam_obs)))
    for i in range(f_samps.shape[0]):
         recon_samps_resampled[i, :] = m_samps[i] / (1 + z_samps[i]) * \
             np.interp(lam_obs,
                       lam0 * (1 + z_samps[i]),
                       f_samps[i,:])

    fig = plt.figure(figsize=(18,4))
    plt.plot(lam_obs, spec_obs, alpha = .5)
    hilo = np.percentile(recon_samps_resampled, [.5, 99.5], axis=0)
    for hl in hilo:
        plt.plot(lam_obs, hl, color='grey', alpha = .5)
    #plt.plot(lam_obs, np.median(recon_samps_resampled, axis=0), linewidth=2, 
    #         label="$E[f^{(obs)} | \mathbf{y}_{photo}]$")
    plt.plot(lam_obs, spec_mod, linewidth=2, color='black', alpha=.8, label="model spec")
    plt.plot(lam_obs, recon_samps_resampled[nearest_z_samp, :], #np.mean(recon_samps_resampled, axis=0)
             linewidth=3, 
             color = sns.color_palette()[2],
             label="$f^{(obs)} | \mathbf{y}_{photo}$ (samp)")
    plt.title("Quasar %s reconstruction from SDSS photo"%qso_info['SDSS_NAME'], fontsize=18)
    plt.ylim(0, spec_obs.max())
    plt.xlim(lam_obs.min(), lam_obs.max())
    plt.xlabel("wavelength $(\AA)$", fontsize=18)
    plt.ylabel("$f(\lambda)$", fontsize=18, rotation='horizontal', labelpad=20)
    plt.legend(fontsize=18)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.savefig(out_dir + "quasar_plots/%s/quasar_%d_mcmc_recon.pdf"%(save_folder, n), bbox_inches='tight')
    plt.close('all')

    # plot pairwise marginals for all variables
    #fig, axarr = plt.subplots(th_samps.shape[1], th_samps.shape[1], figsize=(12,12))
    #for i in range(th_samps.shape[1]):
    #    for j in range(th_samps.shape[1]):
    #        axarr[i,j].scatter(th_samps[(Nsamps/2):,i], th_samps[(Nsamps/2):,j], color=current_palette[2], s=.5)
    #        if i==(th_samps.shape[1]-1):
    #            axarr[i,j].set_xlabel("$z$")
    #        else: 
    #            axarr[i,j].set_xlabel("$w_%d$"%i)
    #        if j==(th_samps.shape[1]-1):
    #            axarr[i,j].set_ylabel("$z$")
    #        else:
    #            axarr[i,j].set_ylabel("$w_%d$"%j)
    #fig.tight_layout()
    #plt.close('all')


##
## Check out marginal values for weights
##
#for n in to_inspect: 






