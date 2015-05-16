import matplotlib.pyplot as plt
import seaborn as sns
import sys, os.path
sys.path.append(
    os.path.abspath('./experiments/galaxy'))
from sample_stamp import load_samples
from glob import glob
import CelestePy.util.infer.mcmc_diagnostics as diagnostics
import CelestePy.util.misc.init_utils as init_utils
from scipy.spatial.distance import pdist, squareform
from CelestePy import gen_model_image, gen_src_image
from CelestePy import SrcParams

def load_mcmc_chains(chain_file_0, num_chains=4, burnin=2500):
    src_samp_chains = []
    ll_samp_chains  = []
    eps_samp_chains = []
    for i in range(num_chains):
        chain_i_file = chain_file_0.replace("chain_0", "chain_%d"%i)
        if not os.path.exists(chain_i_file):
            print "chain_file: %s does not exist"%chain_file_0
            continue
        samp_dict = load_samples(chain_i_file)
        if samp_dict['srcs'][burnin] == samp_dict['srcs'][burnin+1]:
            print "chain_file: %s has zeros?"%chain_file_0
            print samp_dict['srcs'][burnin:(burnin+10)]
            continue

        src_samp_chains.append(samp_dict['srcs'][burnin:,0])
        ll_samp_chains.append(samp_dict['ll'][burnin:])
        eps_samp_chains.append(samp_dict['epsilon'][burnin:])
    return src_samp_chains, ll_samp_chains, eps_samp_chains

def render_extreme_model_images(imgs, samp_chains, eps_chains):
    i, j = find_extreme_samples(samp_chains)
    src_i = SrcParams.init_obj(samp_chains[0][i])
    src_j = SrcParams.init_obj(samp_chains[0][j])
    imgs0 = []
    imgs1 = []
    bands = ['u', 'g', 'r', 'i', 'z']
    for img in imgs:
        img.epsilon = eps_chains[0][i,bands.index(img.band)]
        img0 = gen_model_image([src_i], img)
        imgs0.append(img0)
        img.epsilon = eps_chains[0][j,bands.index(img.band)]
        img1 = gen_model_image([src_j], img)
        imgs1.append(img1)
    return imgs0, imgs1, src_i, src_j

def find_extreme_samples(samp_chains):
    # flatten into numpy array
    fields = ['fluxes', 'theta', 'sigma', 'rho', 'phi']
    samps = np.zeros((len(fields), len(samp_chains[0])))
    for i, field in enumerate(fields):
        params = samp_chains[0][field]
        if len(params.shape)==2:
            params = params[:,2]
        samps[i,:] = params

    # pairwise dists and exterme points
    dists = squareform(pdist(samps.T))
    extreme_inds, _ = np.where(dists==dists.max())
    return extreme_inds[0], extreme_inds[1]

def summarize_univariate_marginals(samp_chains, tex=False):
    """ creates a tex'd table of the statistics for each marginal """
    # compute rhats for each univariate parameter
    r_hat = np.zeros(1, dtype=samp_chains[0].dtype)
    n_eff = np.zeros(1, dtype=samp_chains[0].dtype)
    n_eff_acf = np.zeros(1, dtype=src_samp_chains[0].dtype)
    param_names  = ['fluxes', 'theta', 'phi', 'sigma', 'rho']
    for pname in param_names: 
        param_mat = np.array([schain[pname] for schain in samp_chains])
        if len(param_mat.shape) == 3:
            param_mat = param_mat[:,:,2]
        r_hat[pname] = diagnostics.compute_r_hat(param_mat)
        n_eff[pname] = diagnostics.compute_n_eff(param_mat)
        n_eff_acf[pname] = np.sum( [diagnostics.compute_n_eff_acf(p) for p in param_mat])

    def summarize_param(field, desc, col_ind = None):
        param_mat = np.array([schain[pname] for schain in samp_chains])
        neff = n_eff[0][field]
        rhat = r_hat[0][field]
        neffacf = n_eff_acf[0][field]
        if len(param_mat.shape) == 3:
            param_mat = param_mat[:,:,2]
            neff = neff[2]
            rhat = rhat[2]
            neffacf = neffacf[2]
        if tex:
            out_temp = "  %s & %2.2f & (sd = %2.2f) & ($\hat r = %2.2f$) & ($n_{eff} = %2.2f$) & ($n_{acf} = %2.2f$) \\\\ \n"
        else:
            out_temp = "  %s \t : %2.2f (sd = %2.2f, r_hat = %2.2f, n_eff = %2.2f, n_eff_acf = %2.2f) \n"
        out_string = out_temp%(desc, param_mat.mean(), param_mat.std(), rhat, neff, neffacf)
        return out_string

    if tex:
        summary_string = "\\begin{tabular}{ |l|ccccc } \n "
        summary_string += "  galaxy param & $E[\cdot]$ & & & & \\\\ \n \hline \n"
    else:
        summary_string = ""
    descs = [ "r-flux ($b_r$)         ",
              "theta (prop expo)      ", 
              "phi (angle)            ", 
              "maj-min ratio ($\\rho$)",
              "radius ($\sigma$)      " ]
    for ii,pname in enumerate(param_names): 
        summary_string += summarize_param(pname, descs[ii])
    if tex:
        summary_string += "\\end{tabular}"
    return summary_string

def summarize_pairwise_marginals(samp_chains):
    # flatten all of the chains
    fields = ['fluxes', 'theta', 'sigma', 'rho', 'phi']
    samps = np.zeros((len(fields), len(samp_chains)*len(samp_chains[0])))
    for i, field in enumerate(fields):
        if len(samp_chains[0][field].shape)==2: 
            params = np.row_stack([schain[field] for schain in samp_chains])
        else:
            params = np.concatenate([schain[field] for schain in samp_chains])
        if len(params.shape)==2:
            params = params[:,2]
        samps[i,:] = params

    def mat2table(mat):
        table_str = "\\begin{tabular}{ l|ccccc } \n"
        table_str += " & " + " & ".join(fields) + " \\\\ \n"
        table_str += " \hline \n"
        for i, field in enumerate(fields):
            table_str += "%s & "%field
            table_str += " & ".join(["%2.3f"%c for c in mat[i]])
            table_str += " \\\\ \n"
        table_str += "\\end{tabular}"
        return table_str
    return mat2table(np.corrcoef(samps)), mat2table(np.cov(samps))


if __name__=="__main__":

    # chain files 
    gal_chain_files = glob("run5/gal*chain_0*.bin")
    gal_chain_files.sort()
    gal_chain_files = gal_chain_files[:50]

    texstring = """
\documentclass[12pt]{article}
\\usepackage[a4paper, total={6.5in, 9in}]{geometry}
\\usepackage{graphicx}
\\usepackage{caption}
\\usepackage{subcaption}
\\begin{document}
"""
    f = open('galaxy_summary.tex', 'w')
    f.write(texstring)
    f.close()

    # print out some results 
    for chain in gal_chain_files:
        plt.close("all")
        print "\n============================================"
        print "Galaxy", os.path.basename(chain)

        ## 0. load four chains from disk
        src_samp_chains, ll_samp_chains, eps_samp_chains = \
            load_mcmc_chains(chain, num_chains=4)
        print len(src_samp_chains)

        ## 1.  create table of statistics
        marginal_summary = summarize_univariate_marginals(src_samp_chains, tex=True)
        pairwise_summary, cov_summary = summarize_pairwise_marginals(src_samp_chains)

        ## 2. render extreme samples
        # load in image information for rendering model images
        chain_bname = os.path.basename(chain)        
        stamp_id = chain_bname[16:-12]
        cat_file = "../../data/experiment_stamps/cat-%s.fits"%stamp_id
        cat_srcs, imgs, teff_catalog, us = init_utils.load_imgs_and_catalog([cat_file])
        imgs0, imgs1, src_i, src_j = \
            render_extreme_model_images(imgs, src_samp_chains, eps_samp_chains)

        if not os.path.isdir('figs/%s'%stamp_id):
            os.makedirs('figs/%s'%stamp_id)
        for i in xrange(len(imgs0)):
            vmax = max(imgs0[i].max(), imgs[i].nelec.max())
            vmin = max(imgs0[i].min(), imgs[i].nelec.min())
            def pltimg(arr):
                plt.imshow(arr, origin='lower', 
                                interpolation='none',
                                vmin = vmin,
                                vmax = vmax)
                plt.axis('off')
            pltimg(imgs0[i])
            plt.savefig('figs/%s/img_i_%d'%(stamp_id, i), bbox_inches='tight')
            pltimg(imgs1[i])
            plt.savefig('figs/%s/img_j_%d'%(stamp_id, i), bbox_inches='tight')
            pltimg(imgs[i].nelec)
            plt.savefig('figs/%s/true_img_%d'%(stamp_id, i), bbox_inches='tight')

            # plot residuals
            resid = imgs[i].nelec - imgs0[i]
            zscored = resid / np.sqrt(imgs0[i])
            plt.imshow(zscored, origin='lower', interpolation='none')
            plt.axis('off')
            plt.savefig('figs/%s/resid_%d'%(stamp_id, i), bbox_inches='tight')

        img_idx = [0, 2, 4]
        img_string = "sample 0: ($flux_r = %2.2f, \\theta = %2.2f, \\sigma = %2.2f, \\rho=%2.2f, \\phi=%2.2f$  \\\\ \n" % \
                     (src_i.fluxes['r'], src_i.theta, src_i.sigma, src_i.rho, src_i.phi)
        for i in img_idx:
            img_string += "\\includegraphics[width=.32\\textwidth]{figs/%s/img_i_%d} "%(stamp_id, i)
        img_string += "\\\\ sample 1: ($flux_r = %2.2f, \\theta = %2.2f, \\sigma = %2.2f, \\rho=%2.2f, \\phi=%2.2f$  \\\\ \n" % \
                     (src_j.fluxes['r'], src_j.theta, src_j.sigma, src_j.rho, src_j.phi)
        for i in img_idx:
            img_string += "\\includegraphics[width=.32\\textwidth]{figs/%s/img_j_%d} "%(stamp_id, i)
        img_string += "\\\\ true img: \\\\ \n"
        for i in img_idx:
            img_string += "\\includegraphics[width=.32\\textwidth]{figs/%s/true_img_%d} "%(stamp_id, i)
        img_string += "\\\\ z-scored residual: \\\\ \n"
        for i in img_idx:
            img_string += "\\includegraphics[width=.32\\textwidth]{figs/%s/resid_%d} "%(stamp_id, i)

        ## 3. investigate mixing of chain: 
        plt.close("all")
        for schain in src_samp_chains:
            plt.plot(np.arange(len(schain['fluxes']))[::4], 
                     schain['fluxes'][::4,2])
        plt.savefig("figs/%s/r_flux_trace.pdf"%stamp_id, bbox_inches='tight')

        plt.close("all")
        fields = ['theta', 'sigma', 'rho', 'phi']
        for field in fields:
            for schain in src_samp_chains:
                plt.plot(np.arange(len(schain[field]))[::4],
                         schain[field][::4])
            plt.savefig("figs/%s/%s_trace.pdf"%(stamp_id, field), bbox_inches='tight')
            plt.close("all")
            # compare means/variances for each chain (do they match?)
            # visualize a handful of marginals

        plt.close("all")
        for lchain in ll_samp_chains:
            plt.plot(np.arange(len(lchain))[::4],
                     lchain[::4])
        plt.savefig("figs/%s/ll_trace.pdf"%stamp_id, bbox_inches='tight')
        plt.close("all")

        template = """
\\newpage
\section{Galaxy at %s}
\paragraph{Univariate marginal summaries}
\\begin{center}
%s
\\end{center}

\\paragraph{Bivariate marginal summaries} ~\\\\
\\begin{figure}[h]
    \\centering
    \\begin{subfigure}[b]{0.55\\textwidth}
            %s
            \\caption{correlation}
    \\end{subfigure}
    \\begin{subfigure}[b]{0.55\\textwidth}
            %s
            \\caption{covariance}
    \\end{subfigure}
\\end{figure}

\paragraph{Model Images}
\\begin{center}
%s 
\\end{center}

\paragraph{MCMC Traces}
\\begin{itemize}
\\item r flux: \\\\ \n
%s \n
\\item theta: \\\\ \n
%s \n
\\item sigma: \\\\ \n
%s \n
\\item ll trace: \\\\ \n
%s \n
\\end{itemize}
"""
        template = template%(stamp_id,  
             marginal_summary,
             pairwise_summary, 
             cov_summary, img_string,
             "\\includegraphics[width=.85\\textwidth]{figs/%s/r_flux_trace}"%stamp_id,
             "\\includegraphics[width=.85\\textwidth]{figs/%s/theta_trace}"%stamp_id,
             "\\includegraphics[width=.85\\textwidth]{figs/%s/sigma_trace}"%stamp_id,
             "\\includegraphics[width=.85\\textwidth]{figs/%s/ll_trace}"%stamp_id)

        # write to file (append or overwrite)
        f = open('galaxy_summary.tex', 'a')
        f.write(template)
        f.close()

                # compare flux
        #r_band = np.array([schain['fluxes'][:,2] for schain in src_samp_chains])
        #print "  r_band flux: %2.2f (r hat = %2.2f)"%(r_band.mean(), r_hat[0]['fluxes'][2])

        ## print proportion de vacolours
        #theta_mat = np.array([schain['theta'] for schain in src_samp_chains])
        #print "  Proportion de vacolours: %2.2f (r hat = %2.2f)" % \
        #    (1.0 - theta_mat.mean(), r_hat[0]['theta'])

        #ecc_mat = np.array([schain['rho'] for schain in src_samp_chains])
        #print "  Major minor ratio: %2.2f (r hat = %2.2f)" % \
        #    (ecc_mat.mean(), r_hat[0]['rho'])

        #angle_mat = np.array([schain['phi'] for schain in src_samp_chains])
        #print "  angle: %2.2f (r hat = %2.2f)" % \
        #    (angle_mat.mean() / np.pi * 180., r_hat[0]['phi'])

        #radius_mat = np.array([schain['sigma'] for schain in src_samp_chains])
        #print "  radius: %2.2f (r hat = %2.2f)"%(radius_mat.mean(), r_hat[0]['sigma'])

    f = open('galaxy_summary.tex', 'a')
    f.write("\\end{document}")
    f.close()



    sys.exit()
    ##load four chains
    #src_samp_chains = []
    #for i in range(4):
    #    samp_dict = load_samples("gal_samps_stamp_5.0008-0.4659_chain_%d.bin"%i)
    #    src_samp_chains.append(samp_dict['srcs'][2500:,0])


    # examine one chain
    chain_file = gal_chain_files[2]
    #print chain_file
    #chain_file = "experiments/galaxy/gal_samps_stamp_5.0017-0.4319_chain_0.bin"
    src_samp_chains, ll_samp_chains = load_mcmc_chains(chain_file, num_chains=1, burnin=2500)

    plt.ion()
    fields = ['theta', 'sigma', 'rho', 'phi']
    for field in fields: 
        theta_mat = np.array([schain[field] for schain in src_samp_chains])
        r_hat     = compute_r_hat(theta_mat)
        fig, axarr = plt.subplots(4)
        for i, ax in enumerate(axarr):
            sns.distplot(theta_mat[i,:], ax = ax)
            ax.set_xlim([theta_mat.min(),theta_mat.max()])
            if i==0:
                ax.set_title("%s (rhat = %2.2f)"%(field, r_hat))
        # plt.show()
        plt.savefig("/Users/acm/Dropbox/Proj/astro/DESIMCMC/experiments/galaxy/plots/%s_marginals.pdf"%field, bbox_inches='tight')


    pairs = [['theta', 'sigma'], ['sigma', 'rho']]
    for pair in pairs:

        param_mats = []
        for field in pair:
            param_mats.append(np.array([schain[field] for schain in src_samp_chains]))

        fig, axarr = plt.subplots(2, 2)
        for i, ax in enumerate(axarr.flatten()):
            sns.kdeplot(param_mats[0][i,:], param_mats[1][i,:], ax=ax)
            ax.set_xlim([param_mats[0].min(), param_mats[0].max()])
            ax.set_ylim([param_mats[1].min(), param_mats[1].max()])
            if i == 0: 
                ax.set_title("%s, %s marginal"%(pair[0], pair[1]))

        plt.savefig("/Users/acm/Dropbox/Proj/astro/DESIMCMC/experiments/galaxy/plots/%s_%s_marginals.pdf"%(pair[0], pair[1]), bbox_inches='tight')

    # plot log likelihood chains
    fig = plt.figure()
    means = [lls.mean() for lls in ll_samp_chains]
    for i,lls in enumerate(ll_samp_chains):
        plt.plot(lls[lls != 0], label="chain %d"%i)
    plt.legend()
    plt.show()
    plt.close("all")

    ### compare thetas
    #fig, axarr = plt.subplots(4, 1)
    #for i in range(len(axarr)):
    #    axarr[i].hist(src_samp_chains[i]['theta'], bins=20)
    #plt.show()



    print " comparing autocorrelation"
    from statsmodels.graphics.correlation import plot_corr
    plot_corr(np.corrcoef(samps))

    theta_mat = np.array([schain['theta'] for schain in src_samp_chains])
    plot_acf(theta_mat[1])
    plt.show()

#def plot_samps(src_samps, num_burnin=500):
#    """ plots source samples """
#    samp_dict = load_samples("experiment_cache/samp_cache/gal_samps_stamp_5.0026-0.1581_chain_0.bin")
#    src_samps = samp_dict['srcs'][2500:,0]
#
#    #fig = plt.figure()
#    with sns.axes_style("white"):
#
#        ## plot ANGLE vs RATIO
#        jgrid = sns.jointplot(src_samps['phi'], src_samps['rho'], kind = "kde")
#        jgrid.ax_joint.set_xlabel('$\phi_s$', fontsize=16)
#        jgrid.ax_joint.set_ylabel('$\\rho_s$', fontsize=16)
#
#        ## plot angle vs scale
#        jgrid = sns.jointplot(src_samps['phi'], src_samps['sigma'], kind = "kde")
#        jgrid.ax_joint.set_xlabel('$\phi_s$', fontsize=16)
#        jgrid.ax_joint.set_ylabel('$\sigma_s$', fontsize=16)
#
#        ## plot scale vs ratio
#        jgrid = sns.jointplot(src_samps['sigma'], src_samps['rho'], kind = "kde")
#        jgrid.ax_joint.set_xlabel('$\sigma_s$', fontsize=16)
#        jgrid.ax_joint.set_ylabel('$\\rho_s$', fontsize=16)
#
#        ## plot scale vs TYPE
#        jgrid = sns.jointplot(src_samps['sigma'], src_samps['theta'], kind = "kde")
#        jgrid.ax_joint.set_xlabel('$\sigma_s$', fontsize=16)
#        jgrid.ax_joint.set_ylabel('$\\theta_s$', fontsize=16)


