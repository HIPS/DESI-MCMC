import matplotlib.pyplot as plt
import numpy.random as npr
import cPickle as pickle
import os
import seaborn as sns
sns.set_style("white")
current_palette = sns.color_palette()
npr.seed(42)

exp_types = ["random", "flux", "redshift"]
out_dir   = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/NIPS2015/"
for exp_type in exp_types:

    # load MOG 
    mog_exp_file = "../analysis_mog/%s/results.pkl"%exp_type
    with open(mog_exp_file, 'rb') as handle:
        mog_dict  = pickle.load(handle)
        mog_specs = mog_dict['spec_ids']
        mog_preds = mog_dict['preds']
        mog_true  = mog_dict['z_true']
        mog_per   = mog_dict['preds_per']

    fig = plt.figure(figsize=(8, 8))
    max_z = max(mog_preds.max(), mog_true.max())
    min_z = min(mog_true.min() - .2, 2.2)
    plt.plot([min_z, max_z], [min_z, max_z], linewidth=2, alpha=.5)
    plt_idx = np.arange(0, len(mog_true), 5)
    for n in plt_idx: 
        plt.plot([mog_true[n], mog_true[n]], [mog_per[n,0], mog_per[n,-1]], alpha = .15, color = 'grey', linewidth=1)
    plt.scatter(mog_true[plt_idx], mog_preds[plt_idx], color=current_palette[2], alpha = 1.0)
    plt.xlim(min_z, max_z)
    plt.ylim(min_z, max_z)
    plt.xlabel("$z_{spec}$", fontsize=40, labelpad=20)
    plt.ylabel("$z_{photo}$", fontsize=40, rotation='horizontal', labelpad=40)
    if exp_type != 'random':
        plt.ylabel("")
    plt.tick_params(axis='both', which='major', labelsize=20)
    #plt.title("Posterior expectation model predictions", fontsize=14)
    plt.savefig(os.path.join(out_dir, "scatter_exp-%s.pdf"%exp_type), bbox_inches='tight')




#    z_pred = np.zeros(len(qso_sample_files))
#    z_pred_mode = np.zeros(len(qso_sample_files))
#    z_lo   = np.zeros(len(z_pred))
#    z_hi   = np.zeros(len(z_pred))
#    z_true = np.zeros(len(z_pred))
#    z_lo0 = np.zeros(len(z_pred))
#    z_hi0 = np.zeros(len(z_pred))
#    q_inds = np.zeros(len(z_pred))
#    mode_sample = np.zeros(len(z_pred), dtype=np.int)
#    expected_m = np.zeros(len(z_pred))
#    expected_w = np.zeros((len(z_pred), 4))
#    for i, qso_samp_file in enumerate(qso_sample_files):
#        if i%25==0: print "%d of %d"%(i, len(qso_sample_files))
#        try: 
#            th_samps, lls, q_idx, qso_info, chain_idx = load_redshift_samples(qso_samp_file)
#        except:
#            print "skipping %d"%i
#            continue
#        q_inds[i] = q_idx
#
#        Nsamps = th_samps.shape[0]
#        # compare predict to true
#        z_true[i]        = qso_info['Z_VI']
#        z_pred[i]        = th_samps[Nsamps/2:, 0].mean()
#        z_lo[i], z_hi[i] = np.percentile(th_samps[Nsamps/2:, 0], [.5, 99.5])
#        z_lo0[i], z_hi0[i] = np.percentile(th_samps[Nsamps/2:, 0], [5, 95])
#
#        # kernel density estimate to find the highest mode
#        z_unique = np.unique(th_samps[Nsamps/2:, 0])
#        density  = kde.gaussian_kde(z_unique, bw_method = .08 ) #'silverman')
#        mode_sample[i] = density(z_unique).argmax()
#        z_pred_mode[i] = z_unique[ mode_sample[i] ]
#
#        # expected magnitude/
#        expected_m[i] = np.exp(th_samps[Nsamps/2:, -1]).mean()
#
#        # expected weights
#        ws = np.exp(th_samps[Nsamps/2:, 1:-1])
#        ws /= np.sum(ws, axis=1, keepdims=True)
#        expected_w[i, :] = ws.mean(axis=0)

