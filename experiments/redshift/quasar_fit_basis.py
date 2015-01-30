#
# This script implements functions for maximum likelihood 
# estimation of a basis for a group of Quasar Spectra.  
#
# Roughly, this procedure is the following:
#   - Resample spectra from lam_obs into rest frame grid lam0, (using Z_spec)
#   - Fit optimize basis and weights in an NMF-like framework (with normal errors)
#
import fitsio
import numpy as np
import numpy.random as npr
import numpy as npp
from scipy import interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from funkyyak import grad, numpy_wrapper as np
npr.seed(42)
from scipy.optimize import minimize
from redshift_utils import load_data_clean_split, project_to_bands, sinc_interp, \
                           check_grad, fit_weights_given_basis, \
                           evaluate_random_direction, ParamParser, \
                           resample_rest_frame
from slicesample import slicesample
import GPy
import os
import cPickle as pickle

def save_basis_fit(th, lam0, lam0_delta, parser):
    """ save basis fit info """
    # grab B value for shape info
    B = parser.get(th, 'betas')
    with open('cache/basis_fit_K-%d_V-%d.pkl'%B.shape, 'wb') as handle:
        pickle.dump(th, handle)
        pickle.dump(lam0, handle)
        pickle.dump(lam0_delta, handle)
        pickle.dump(parser, handle)

def load_basis_fit(fname):
    with open(fname, 'rb') as handle:
        th         = pickle.load(handle)
        lam0       = pickle.load(handle)
        lam0_delta = pickle.load(handle)
        parser     = pickle.load(handle)
    return th, lam0, lam0_delta, parser

def make_functions(X, inv_var, lam0, lam0_delta, K, K_chol, Kinv_beta, sig2_omega, sig2_mu):
    parser = ParamParser()
    V      = len(lam0)
    N      = X.shape[0]
    parser.add_weights('mus', (N, 1))
    parser.add_weights('betas', (K, V))
    parser.add_weights('omegas', (N, K))

    ## weighted loss function - observations have gaussian noise
    #def loss_fun(th_vec, X, inv_var, lam0_delta, K):
    def loss_fun(th_vec):
        """ Negative log likelihood function.  The likelihood model encoded here is

                beta_k  ~ GP(0, K)
                omega_k ~ Normal(0, 1)
                mu_k    ~ Normal(0, 10)

          Normalize Basis and weights so they both sum to 1
                B_k = exp(beta_k) / sum( exp(beta_k) DeltaLam)
                w_k = exp(w_k) / sum(exp(w_i))
                m   = exp(mu_k)
                f   = m \sum w_k B_k

          Observations are normal about the latent spectra, with known variance
                X_lam ~ Normal(f, var_lam)
        """
        # unpack params
        N      = X.shape[0]
        mus    = parser.get(th_vec, 'mus')
        betas  = parser.get(th_vec, 'betas')
        omegas = parser.get(th_vec, 'omegas')

        # exponentiate and normalize params
        W = np.exp(omegas)
        W = W / np.sum(W, axis=1, keepdims=True)
        B = np.exp(np.dot(K_chol, betas.T).T)
        B = B / np.sum(B * lam0_delta, axis=1, keepdims=True)
        M = np.exp(mus)
        Xtilde = np.dot(W*M, B)
        return np.nansum( inv_var * np.square(X - Xtilde) )
    loss_grad = grad(loss_fun)

    ## joint prior over parameters
    def prior_loss(th, Kinv_beta = Kinv_beta):
        """ Prior over weights and basis.
            - th_mat    : K x (N + V) matrix holding all weights and basis params
            - N         : number of examples in training set
            - Kinv_beta : Inverse covariance of beta (log basis)
                          functions (for encoding smooth and spikes)
            - sig2_omega: prior variance on log weights
        """
        mus    = parser.get(th, 'mus')
        betas  = parser.get(th, 'betas')
        omegas = parser.get(th, 'omegas')
        loss_mus    = .5 / (sig2_mu) * np.sum(np.square(mus))
        loss_omegas = .5 / (sig2_omega) * np.sum(np.square(omegas))
        #loss_betas  = .5 * np.sum(np.dot(betas, Kinv_beta) * betas)
        loss_betas  = .5 * np.sum(np.square(betas))
        return loss_omegas + loss_mus + loss_betas 
    prior_loss_grad = grad(prior_loss)
    return parser, loss_fun, loss_grad, prior_loss, prior_loss_grad


## simple gradient based NMF w/ gaussian noise training function
def train_model(th, loss_fun, loss_grad, prior_loss, prior_grad, 
                cvx_iter=20000, sgd_iter = 5000,
                learning_rate = 1e-5, momentum = .9, verbose=100):


    ## get to a reasonable spot w/ CVX OPT
    #res = minimize(fun = lambda(th): loss_fun(th) + prior_loss(th), 
    #               jac = lambda(th): loss_grad(th) + prior_grad(th),
    #               x0  = th, 
    #               method = 'BFGS',
    #               options = {'gtol':1e-6, 'disp':True, 'maxiter':10})
    #th = res.x

    # dynamically set learning rate heuristically
    #initial_rate = learning_rate

    # gradient descent + momentum
    #cur_dir = np.zeros(th.shape)
    #lls     = np.zeros(sgd_iter)
    #print "{0:15}|{1:15}|{2:15}|{3:15}|{4:15}".format(
    #    "  Iter   ", 
    #    "  Train Err ", 
    #    "  Objective Fun ", 
    #    "  Step_size  ", 
    #    "  Grad Mag   ")
    #for epoch in range(sgd_iter):

    #    # compute gradient of objective function, reset learning rate
    #    grad_th  = loss_grad(th) + prior_grad(th)
    #    grad_mag = np.sqrt(np.sum(grad_th*grad_th))
    #    learning_rate = 1. / grad_mag

    #    # compute step direction, accounts for momentum
    #    cur_dir    = momentum * cur_dir + (1.0 - momentum) * grad_th
    #    step_mag   = np.sqrt(np.sum(np.square(learning_rate * cur_dir)))

    #    # take a step
    #    step_vec   = .01 / grad_mag * cur_dir
    #    step_mag   = np.sqrt(np.sum(step_vec * step_vec))

    #    th        -= step_vec
    #    lls[epoch] = loss_fun(th)

    #    # make sure it's not nan
    #    #while np.isnan(lls[epoch]): 
    #    #    th            += learning_rate * cur_dir   # undo step
    #    #    #learning_rate /= 2
    #    #    #th            -= learning_rate * cur_dir
    #    #    lls[epoch] = loss_fun(th)
    #    #learning_rate = initial_rate

    #    if epoch % verbose == 0:
    #        print "{0:15}|{1:15}|{2:15}|{3:15}|{4:15}".format(
    #            epoch, 
    #            "%.12g"%lls[epoch],
    #            "%.12g"%(lls[epoch] + prior_loss(th)),
    #            "%2.4f"%step_mag, 
    #            "%2.4f"%grad_mag)

    #### Optimize w/ LBFGS with NO PRIOR 
    print "Switching to CVX Optimization Method"
    th_shape = th.shape
    res = minimize(fun = lambda(th): loss_fun(th) + prior_loss(th),
                   jac = lambda(th): loss_grad(th) + prior_grad(th),
                   x0  = th, 
                   method = 'L-BFGS-B',
                   options = {'gtol':1e-6, 'ftol':1e-6, 'disp':True, 'maxiter':cvx_iter})
    th = res.x

    # go from whitened space to GP space
    betas = parser.get(th, 'betas')
    parser.set(th, 'betas', np.dot(K_chol, betas.T).T)
    return th

if __name__=="__main__":

    ## load a handful of quasar spectra
    lam_obs, qtrain, qtest = \
        load_data_clean_split(spec_fits_file = 'quasar_data.fits', 
                              Ntrain = 400)

    ## first find a positive decomposition of quasar spectra on training data
    quasar_zerr    = qtrain['Z_err']
    N              = qtrain['spectra'].shape[0]

    ## initialize a basis using existing eigenQuasar Model
    lam_subsample = 10
    header     = fitsio.read_header('../../data/eigen_specs/spEigenQSO-55732.fits')
    eigQSOfits = fitsio.FITS('../../data/eigen_specs/spEigenQSO-55732.fits')
    lam0       = 10.**(header['COEFF0'] + np.arange(header['NAXIS1']) * header['COEFF1'])
    lam0       = lam0[::lam_subsample]
    lam0_delta = np.concatenate((lam0[1:] - lam0[:-1], [lam0[-1] - lam0[-2]]))
    eigQSO     = eigQSOfits[0].read()[:, ::lam_subsample]
    K          = eigQSO.shape[0]

    # resample spectra and spectra inverse variance onto common rest frame
    spectra_resampled, spectra_ivar_resampled, lam_mat = \
        resample_rest_frame(qtrain['spectra'], 
                            qtrain['spectra_ivar'],
                            qtrain['Z'], 
                            lam_obs, 
                            lam0)

    ## construct smooth + spiky prior over betas
    beta_kern = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=40)
    K_beta = beta_kern.K(lam0.reshape((-1, 1)))
    K_chol = np.linalg.cholesky(K_beta)
    K_inv  = np.linalg.inv(K_beta)

    ## clean nans in data and inverse variance 
    X                  = spectra_resampled
    X[np.isnan(X)]     = 0
    Lam                = spectra_ivar_resampled
    Lam[np.isnan(Lam)] = 0

    ## set up the likelihood and prior functions
    parser, loss_fun, loss_grad, prior_loss, prior_loss_grad  = \
        make_functions(X, Lam, lam0, lam0_delta, K, 
                       Kinv_beta  = K_inv, 
                       K_chol     = K_chol,
                       sig2_omega = 1., 
                       sig2_mu    = 10.)

    ## initialize basis and weights
    basis_cache = 'cache/basis_fit_K-4_V-1364.pkl'
    USE_CACHE = True
    if os.path.exists(basis_cache) and USE_CACHE:
        th, lam0, lam0_delta, parser = load_basis_fit(basis_cache)
    else:
        th = np.zeros(parser.N)
        parser.set(th, 'betas', .01 * K_chol.dot(np.random.randn(len(lam0), K)).T)
        parser.set(th, 'omegas', .01 * npr.randn(N, K))
        parser.set(th, 'mus', .01 * npr.randn(N))

    ## sanity check gradient
    check_grad(fun = lambda th: loss_fun(th) + prior_loss(th), # X, Lam), 
               jac = lambda th: loss_grad(th) + prior_loss_grad(th), #, X, Lam),
               th  = th)

    ## train model
    parser.set(th, 'betas', np.linalg.solve(K_chol, parser.get(th, 'betas').T).T)
    th = train_model(th, loss_fun, loss_grad, prior_loss, prior_loss_grad,
                          cvx_iter=50000, sgd_iter=10,
                          learning_rate = 1e-4, momentum=.9, verbose=10)
    print "Fit loss: ", loss_fun(th)
    dth = loss_grad(th)
    print "Gradient mag: ", np.sqrt((dth*dth).sum())
    mus    = parser.get(th, 'mus')
    betas  = parser.get(th, 'betas')
    omegas = parser.get(th, 'omegas')
    W = np.exp(omegas)
    W = W / np.sum(W, axis=1, keepdims=True)
    B = np.exp(betas)
    B = B / np.sum(B * lam0_delta, axis=1, keepdims=True)
    M = np.exp(mus)
    Xtilde = np.dot(W*M, B)

    # cache result
    save_basis_fit(th, lam0, lam0_delta, parser)

    ## sanity check out of sample fits
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



## or load from file load and interpolate existing basis
#def load_initial_values(lam0, num_quasars=N):
#    th_init  = np.load("cache/basis_th_K-4_V-1364.npy")
#    lam_init = np.load("cache/lam0_V-1364.npy")
#    omegas   = th_init[:, 0:num_quasars].T
#    betas    = th_init[:, num_quasars:]
#    betas_resamp = np.zeros((betas.shape[0], len(lam0)))
#    for k in range(B.shape[0]):
#        betas_resamp[k, :] = np.interp(x = lam0, 
#                                       xp = lam_init, 
#                                       fp = betas[k, :], 
#                                       left = 0., right = 0.)
#    return np.column_stack((omegas.T, betas_resamp))
#th = load_initial_values(lam0)
#print loss_fun(th)


## DEAD CODE
## iteratively minimize weight/basis pairs
#def train_iterative(th, X, Lam, max_iter=100): 
#    def loss_omegas(omegas, B):
#        ll_omega = 1 / (100.) * np.sum(np.square(omegas))
#        Xtilde   = np.dot(np.exp(omegas), B)
#        return np.sum( Lam * np.square(X - Xtilde) ) + ll_omega
#    loss_omegas_grad = grad(loss_omegas)
#
#    def loss_betas(betas, W):
#        #ll_beta = np.sum(np.dot(np.dot(betas, KB_inv), betas.T))
#        ll_beta = np.sum(np.square(betas))
#        B = np.exp(betas)
#        B = B / np.sum(B, axis = 1, keepdims=True)
#        Xtilde = np.dot(W, np.exp(betas))
#        return np.sum( Lam * np.square(X - Xtilde)) + ll_beta
#    loss_betas_grad = grad(loss_betas)
#
#    #### minimize w/ convex opt method
#    print "    Iter       |    Train err  |   grad_mag  "
#    N = X.shape[0]
#    omegas = th[:, 0:N]
#    betas  = th[:, N:]
#    curr_loss = loss_fun(np.column_stack((omegas, betas)), X, Lam)
#    lls = []
#    for iter_i in range(max_iter):
#
#        ## fix weights and minimize w.r.t basis
#        W = np.exp(omegas).T
#        res = minimize( x0 = betas.ravel(),
#                        fun = lambda b: loss_betas(b.reshape(betas.shape), W),
#                        jac = lambda b: loss_betas_grad(b.reshape(betas.shape), W).ravel(),
#                        method = 'L-BFGS-B',
#                        options = {'gtol':1e-6, 'disp':False, 'maxiter':30})
#        betas = res.x.reshape(betas.shape)
#
#        ## debug compute marignal loss
#        print " post-basis opt loss: ", loss_fun(np.column_stack((omegas, betas)), X, Lam)
#
#        ## now fix basis and minimize w.r.t. weights
#        B = np.exp(betas)
#        B = B / np.sum(B, axis=1, keepdims=True)
#        res = minimize( x0 = omegas.ravel(),
#                        fun = lambda o: loss_omegas(o.reshape(omegas.shape).T, B),
#                        jac = lambda o: loss_omegas_grad(o.reshape(omegas.shape).T, B).ravel(),
#                        method = 'L-BFGS-B',
#                        options = {'gtol':1e-6, 'disp':False, 'maxiter':30})
#        omegas = res.x.reshape(omegas.shape)
#
#        ## compute marginal loss
#        th = np.column_stack((omegas, betas))
#        lls.append(loss_fun(th, X, Lam))
#        th_grad = loss_grad(th, X, Lam)
#        print "{0:15}|{1:15}|{2:15}".format(iter_i, "%5g"%lls[-1], np.sqrt(np.sum(th_grad*th_grad)))
#
#    th_fine = np.column_stack((omegas, betas))
#    return th_fine, lls



