import fitsio
import numpy as np
import numpy.random as npr
from scipy.optimize import minimize
from scipy import interpolate
from funkyyak import grad, numpy_wrapper as np
from redshift_utils import load_data_clean_split, project_to_bands, fit_weights_given_basis
from slicesample import slicesample
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
from quasar_infer_photometry import pixel_likelihood
sns.set_style("white")
current_palette = sns.color_palette()
npr.seed(42)

if __name__=="__main__":

    out_dir = "/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/figs/"

    ## load a handful of quasar spectra
    lam_obs, qtrain, qtest = \
        load_data_clean_split(spec_fits_file = 'quasar_data.fits',
                              Ntrain         = 400)

    ## load in basis
    basis_string = "13637"                # size of the basis
    th   = np.load("cache/basis_th_K-4_V-%s.npy"%basis_string)
    lls  = np.load("cache/lls_K-4_V-%s.npy"%basis_string)
    lam0 = np.load("cache/lam0_V-%s.npy"%basis_string)
    N    = th.shape[1] - lam0.shape[0]
    omegas = th[:,:N]
    betas  = th[:, N:]
    W = np.exp(omegas)
    B = np.exp(betas)
    B = B / B.sum(axis=1, keepdims=True)

    ## map inference for each quasar
    Nqso = qtest['spectra'].shape[0]
    z_maps = np.zeros(Nqso)
    for n in range(Nqso):

        ## fit w's and red-shift w/ MCMC
        print "   ... map %d of %d "%(n, Nqso)
        spec_n             = qtest['spectra'][n, :]
        spec_n[spec_n < 0] = 0
        spec_ivar_n        = qtest['spectra_ivar'][n, :]
        z_n                = qtest['Z'][n]
        w_n                = fit_weights_given_basis(B, lam0, spec_n, spec_ivar_n, z_n, lam_obs)
        #mu_n               = project_to_bands(np.atleast_2d(spec_n), lam_obs)
        mu_n               = project_to_bands(np.atleast_2d(w_n.dot(B)), lam0)
        x_n                = npr.poisson(mu_n).ravel()
        #w_n = np.random.rand(len(w_n))
        if False:
            plt.plot(lam_obs, spec_ivar_n, alpha=.5, color='grey')
            plt.plot(lam_obs, spec_n, label="noisy obs spec")
            plt.plot(lam0*(1+z_n), w_n.dot(B), label="fit mean")
            plt.legend()
            plt.show()

        ## do maximum likelihood using numerical differences
        lnfun = lambda th: -pixel_likelihood(th[-1], th[:-1], x_n, lam0, B)
        def lnjac(th): 
            dth = np.zeros(len(th))
            for i in range(len(th)):
                de = np.zeros(len(th))
                de[i] = 1e-5
                dth[i] = (lnfun(th + de) - lnfun(th - de)) / (2*1e-5)
            return dth

        #th0 = np.concatenate((w_n, [z_n]))
        th0 = np.concatenate([.0001*np.random.rand(4), [.5]])
        res = minimize(x0 = th0,
                       fun = lnfun,
                       #jac = lnjac,
                       method = 'L-BFGS-B', 
                       #method = 'TNC',
                       #method = 'SLSQP',
                       bounds = [(0, None)]*len(th0))
        z_maps[n] = res.x[-1]
        print res.x
        print "    true = %2.4f, pred = %2.4f"%(qtest['Z'][n], z_maps[n])
        print " result less than GT: ", lnfun(res.x) < lnfun(np.concatenate((w_n, [z_n])))
        print weights_to_bands(res.x[-1], res.x[:-1], x_n, lam0, B)
        print mu_n
        print lnfun(res.x)
        print lnfun(np.concatenate((w_n, [z_n])))

        # now 
        if False: 
            w_n_hat = res.x[:-1]
            z_n_hat = res.x[-1]
            spec_n_hat = w_n.dot(B)  # rest frame
            lam_n_hat = lam0 * (1 + z_n_hat)
            plt.plot(lam_obs, qtest['spectra'][n,:], label = "observed ($z = %2.2f$)"%z_n)
            plt.plot(lam_n_hat, spec_n_hat, label = "photo_spec ($\hat z = %2.2f$)"%z_n_hat)
            plt.legend()
            plt.show()

    fig = plt.figure(figsize=(6, 6))
    max_z = max(qtest['Z'].max(), z_maps.max()) + .2
    min_z = -.2
    plt.scatter(qtest['Z'], z_maps, color=current_palette[2])
    plt.plot([min_z, max_z], [min_z, max_z], linewidth=2, alpha=.5)
    #for n in range(len(z_pred)):
    #    plt.plot([qtest['Z'][n], qtest['Z'][n]], [z_lo[n], z_hi[n]], alpha = .5, color = 'grey')
    plt.xlim(min_z, max_z)
    plt.ylim(min_z, max_z)
    plt.xlabel("full spec measurment, $z_{spec}$", fontsize=14)
    plt.ylabel("photometric measurement, $z_{photo}$", fontsize=14)
    plt.title("Posterior expectation model predictions", fontsize=14)
    plt.savefig(out_dir + "map_z_preds_K-%d_V-%d.pdf"%B.shape, bbox_inches='tight')
