import sys, os
from glob import glob
import fitsio
import cPickle as pickle
import numpy as np
import numpy.random as npr
from CelestePy.util.misc import check_grad
from CelestePy.util.infer.optimizers import *
from scipy.optimize import minimize
sys.path.append('../../')
import redshift_utils   as ru
import quasar_fit_basis as qfb
import GPy

###
### Experiment Params
###
NUM_TRAIN_EXAMPLE = 20000
MAX_LBFGS_ITER    = 20000
MAX_RMSPROP_ITER  = 500
SEED              = 42
NUM_BASES         = 4
BETA_VARIANCE     = 1.
BETA_LENGTHSCALE  = 40.

###
### train and plot functions
###
def plot(th):
    if False:
        def plot_fits(qobj, idxs, W, B, refit_w = True, sgd_iter=1000):
            for n in idxs:
                fig, axarr = plt.subplots(2, 1, figsize=(12,6))
                z_n    = qobj['Z'][n]
                spec_n = qobj['spectra'][n, :]
                ivar_n = qobj['spectra_ivar'][n, :]
                if refit_w:
                    w = fit_weights_given_basis(B, lam0, spec_n, ivar_n, z_n, lam_obs, sgd_iter=sgd_iter)
                else:
                    w = W[n, :]
                    m = M[n]
                axarr[0].plot(lam_obs / (1 + z_n), spec_n)
                axarr[0].plot(lam0, m*w.dot(B), linewidth=3)
                axarr[0].plot(lam_obs / (1 + z_n), np.sqrt(ivar_n), color='grey', alpha=.5)
                axarr[0].set_xlim( min(lam_obs/(1+z_n)), max(lam_obs/(1+z_n)) )
                axarr[1].bar(np.arange(len(w)), w, width=.7, alpha=.5)
            plt.show()

        train_idxs = [0, 20, 14, 300]
        plot_fits(qtrain, train_idxs, W, B, refit_w = False)
        plot_fits(qtrain, train_idxs, W, B, refit_w =True)

        ## visualize one example in a random direction
        n = 20
        w,lfun = fit_weights_given_basis(B, lam0, qtrain['spectra'][n],
                                             qtrain['spectra_ivar'][n],
                                             qtrain['Z'][n], 
                                             lam_obs, return_loss=True, sgd_iter=1000)
        ll_rand = evaluate_random_direction(fun = lfun, x0=np.log(w), n = 100, delta=.01)
        fig = plt.figure()
        plt.plot(ll_rand)
        plt.title(" Random direction for fit_weights_given_basis loss")
        plt.show()
        test_idxs = [0, 10, 95, 139]
        plot_fits(qtest, test_idxs, refit_w = True)


if __name__=="__main__":

    # DR10 qso dataset and spec files
    qso_psf_flux, qso_z, spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = "random")

    ## randomly subselect NUM_TRAIN
    np.random.seed(SEED)
    rand_idx      = np.random.permutation(len(train_idx))
    train_idx_sub = train_idx[rand_idx[0:NUM_TRAIN_EXAMPLE]]

    #rand_idx      = np.random.permutation(len(test_idx))
    #test_idx_sub  = test_idx[rand_idx[0:NUM_TEST_EXAMPLE]]

    ## only load in NUM_TRAIN spec files
    train_spec_files = np.array(spec_files)[train_idx_sub]
    #test_spec_files  = np.array(spec_files)[test_idx]
    spec_grid, spec_ivar_grid, spec_mod_grid, unique_lams, spec_zs, spec_ids, badids = \
        ru.load_specs_from_disk(train_spec_files)

    # cache 
    with open("qso_spec_data.bin", 'wb') as handle:
        np.save(handle, spec_grid)
        np.save(handle, spec_ivar_grid)
        np.save(handle, spec_mod_grid)
        np.save(handle, unique_lams)
        np.save(handle, spec_zs)

    ## initialize a basis using existing eigenQuasar Model
    lam_subsample = 5
    lam0, lam0_delta = ru.get_lam0(lam_subsample=lam_subsample,
            eigen_file = '../../../../data/eigen_specs/spEigenQSO-55732.fits')

    # resample spectra and spectra inverse variance onto common rest frame
    spectra_resampled, spectra_ivar_resampled, lam_mat = \
        ru.resample_rest_frame(spectra      = spec_grid,
                               spectra_ivar = spec_ivar_grid, 
                               zs           = spec_zs,
                               lam_obs      = unique_lams, 
                               lam0         = lam0)

    ## construct smooth + spiky prior over betas
    print "   Computing covariance cholesky "
    beta_kern = GPy.kern.Matern52(input_dim   = 1,
                                  variance    = BETA_VARIANCE,
                                  lengthscale = BETA_LENGTHSCALE)
    K_beta = beta_kern.K(lam0.reshape((-1, 1)))
    K_chol = np.linalg.cholesky(K_beta)

    ## clean nans in data and inverse variance 
    X                  = spectra_resampled
    X[np.isnan(X)]     = 0
    Lam                = spectra_ivar_resampled
    Lam[np.isnan(Lam)] = 0

    ## set up the likelihood and prior functions
    parser, loss_fun, loss_grad, prior_loss, prior_loss_grad  = \
        qfb.make_functions(X, Lam, lam0, lam0_delta,
                           K          = NUM_BASES,
                           K_chol     = K_chol,
                           sig2_omega = 1.,
                           sig2_mu    = 10.)

    ## initialize basis and weights
    Nspec, Vspec = spectra_resampled.shape
    th = np.zeros(parser.N)
    parser.set(th, 'betas', .01 * K_chol.dot(np.random.randn(Vspec, NUM_BASES)).T)
    parser.set(th, 'omegas', .01 * npr.randn(Nspec, NUM_BASES))
    parser.set(th, 'mus', .01 * npr.randn(Nspec))

    ### if file exists, at least pull basis out
    #if os.path.exists("basis_fit_K-%d_V-2728.pkl"%NUM_BASES):
    #    print "basis fit file exists! - checking basis"
    #    th_disk, lam0_disk, lam0_delta_disk, parser_disk = \
    #        qfb.load_basis_fit("basis_fit_K-%d_V-2728.pkl"%NUM_BASES)
    #    if np.all(lam0_disk == lam0) and :
    #        print "basis size is consistent - initializing..."
    #        parser.set(th, 'betas', parser_disk.get('betas'))

    ## sanity check gradient
    check_grad(fun = lambda th: loss_fun(th) + prior_loss(th), # X, Lam), 
               jac = lambda th: loss_grad(th) + prior_loss_grad(th), #, X, Lam),
               th  = th)

    obj_vals = []
    min_val = np.inf
    min_x   = None
    def callback(x, i, g): 
        global min_val
        global min_x
        global obj_vals
        if i % 10 == 0:
            loss_val = loss_fun(x) + prior_loss(x)
            if loss_val < min_val:
                min_x = x
                min_val = loss_val
            print " %d, loss = %2.4g, grad = %2.4g " % \
                (i, loss_val, np.sqrt(np.dot(g,g)))
            obj_vals.append(loss_val)

    ## train with rms_prop
    out = rms_prop(grad      = lambda th: loss_grad(th) + prior_loss_grad(th),
                   x         = th,
                   callback  = callback,
                   num_iters = MAX_RMSPROP_ITER,
                   step_size = .01,
                   gamma     = .98)
    qfb.save_basis_fit(min_x, lam0, lam0_delta, parser, data_dir="")

    ## tighten it up a bit
    res = minimize(fun = lambda th: loss_fun(th) + prior_loss(th),
                   jac = lambda th: loss_grad(th) + prior_loss_grad(th),
                   x0  = min_x,
                   method = 'L-BFGS-B',
                   options = {'maxiter':MAX_LBFGS_ITER, 'disp':True})

    ## save result
    qfb.save_basis_fit(res.x, lam0, lam0_delta, parser, data_dir="")
    th_mle = res.x
    ll_mle = loss_fun(th_mle) + prior_loss(th_mle)

    ## plot random profiles
    #plot a handful of random directions
    egrid = np.linspace(-1.5, 1.5, 30)
    def gen_random_profile(th): 
        rdir     = npr.randn(th.shape[0])
        rdir     = rdir / np.sqrt(np.sum(rdir**2))
        rdirloss = lambda(e): loss_fun(th + e*rdir) + prior_loss(th + e*rdir)
        lgrid = np.array([rdirloss(e) for e in egrid])
        return lgrid

    plt.figure(1)
    for i in range(5):
        lprof = gen_random_profile(th_mle)
        plt.plot(egrid, lprof)
    plt.vlines(x = 0, ymin = lprof.min(), ymax = lprof.max(), linewidth=4)
    plt.title("objective function, random directions")
    plt.savefig("obj_fun_dirs.pdf", bbox_inches='tight')
    plt.close("all")

