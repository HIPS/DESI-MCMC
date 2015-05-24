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
SPLIT_TYPE        = "redshift"  #split_types = ["random", "flux", "redshift"]
NUM_TRAIN_EXAMPLE = 20000
MAX_LBFGS_ITER    = 4000
MAX_RMSPROP_ITER  = 2000
RMSPROP_STEP      = .05
RMSPROP_MOMENTUM  = .99
SEED              = 42
NUM_BASES         = 4
BETA_VARIANCE     = 1.
BETA_LENGTHSCALE  = 40.

CACHE_TRAIN_FILE = "qso_spec_data.bin"
def load_cached_train_matrix(train_spec_files, train_idx, force_no_cache=False):
    # check if cached file exists and is legit
    if os.path.exists(CACHE_TRAIN_FILE) and not force_no_cache:
      handle    = open(CACHE_TRAIN_FILE, 'rb')
      train_idx_disk = np.load(handle)
      # confirm input train_idx from script matches train_idx from disk
      if np.all(train_idx_disk == train_idx):
          print "Found matching cached qso_spec_data matrix on disk!"
          spec_grid      = np.load(handle)
          spec_ivar_grid = np.load(handle)
          spec_mod_grid  = np.load(handle)
          unique_lams    = np.load(handle)
          spec_zs        = np.load(handle)
          spec_ids       = np.load(handle)
          return spec_grid, spec_ivar_grid, spec_mod_grid, unique_lams, spec_zs, spec_ids

    #### load the slow way :(
    print "cached training matrix is not the same!!! loading from spec files! (this will take a while)"
    spec_grid, spec_ivar_grid, spec_mod_grid, unique_lams, spec_zs, spec_ids, badids = \
        ru.load_specs_from_disk(train_spec_files)
    with open(CACHE_TRAIN_FILE, 'wb') as handle:
        np.save(handle, train_idx)
        np.save(handle, spec_grid)
        np.save(handle, spec_ivar_grid)
        np.save(handle, spec_mod_grid)
        np.save(handle, unique_lams)
        np.save(handle, spec_zs)
        np.save(handle, spec_ids)
    return spec_grid, spec_ivar_grid, spec_mod_grid, unique_lams, spec_zs, spec_id

def minimize_chunk(fun, jac, x0, method, chunk_size=250):
    """ minimize function that saves every few iterations """
    num_chunks = int(MAX_LBFGS_ITER / float(chunk_size)) + 1
    for chunk_i in range(num_chunks):
        print "optimizing chunk %d of %d (curr_ll = %2.5g)"%(chunk_i, num_chunks, fun(x0))
        res = minimize(fun = fun, jac = jac, x0 = x0, method = method,
                       options = {'maxiter': chunk_size, 'disp': True})
        x0  = res.x
        qfb.save_basis_fit(res.x, lam0, lam0_delta, parser, data_dir="")
    return res

def minibatch_minimize(grad, num_epochs=100, batch_size=250,
                        callback=None, step_size=0.1, mass=0.9, eps=1e-8):
    avg_sq_grad = np.ones(len(x)) # Is this really a sensible initialization?
    for epoch in xrange(num_epochs):

        # full grad once an epoch (this might be slow)
        g = grad(x)
        if callback: callback(x, epoch, g)

        # divide up
        N_example   = parser.get(th_vec, 'mus').shape[0]
        idxs        = npr.permutation(N_example)
        num_batches = int(N_example / float(batch_size))+1
        for batch_i in xrange(num_batches):
            # grab minibatch
            starti    = batch_i * batch_size
            endi      = min(starti + batch_size, N_example)
            batch_idx = idxs[starti:endi]

            g = grad(x, batch_idx)
            avg_sq_grad = avg_sq_grad * mass + g**2 * (1 - mass)
            x -= step_size * g/(np.sqrt(avg_sq_grad) + eps)           
    return x

if __name__=="__main__":

    # DR10 qso dataset and spec files
    qso_psf_flux, qso_z, spec_files, train_idx, test_idx = \
        ru.load_DR10QSO_train_test_idx(split_type = SPLIT_TYPE)

    ## randomly subselect NUM_TRAIN
    np.random.seed(SEED)
    rand_idx      = np.random.permutation(len(train_idx))
    train_idx_sub = train_idx[rand_idx[0:NUM_TRAIN_EXAMPLE]]

    #rand_idx      = np.random.permutation(len(test_idx))
    #test_idx_sub  = test_idx[rand_idx[0:NUM_TEST_EXAMPLE]]

    ## only load in NUM_TRAIN spec files
    train_spec_files = np.array(spec_files)[train_idx_sub]
    #test_spec_files  = np.array(spec_files)[test_idx]

    spec_grid, spec_ivar_grid, spec_mod_grid, unique_lams, spec_zs, spec_ids = \
         load_cached_train_matrix(train_spec_files, train_idx)

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
        if i % 1 == 0:
            loss_val = loss_fun(x) + prior_loss(x)
            if loss_val < min_val:
                min_x = x
                min_val = loss_val
            print " %d, loss = %2.4g, grad = %2.4g " % \
                (i, loss_val, np.sqrt(np.dot(g,g)))
            obj_vals.append(loss_val)

    ## train with rms_prop
    def full_loss_grad(th, idx=None):
        return loss_grad(th, idx) + prior_loss_grad(th, idx)

    minibatch_minimize(grad       = full_loss_grad,
                       x          = th,
                       callback   = callback,
                       num_epochs = 10,
                       batch_size = 24,
                       step_size  = RMSPROP_STEP,
                       mass       = RMSPROP_MOMENTUM)
    qfb.save_basis_fit(min_x, lam0, lam0_delta, parser, data_dir="")

    ## tighten it up a bit
    res = minimize_chunk(fun = lambda th: loss_fun(th) + prior_loss(th),
                         jac = lambda th: loss_grad(th) + prior_loss_grad(th),
                         x0  = min_x,
                         method = 'L-BFGS-B')

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

