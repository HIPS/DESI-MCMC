import fitsio
import numpy as np
import numpy.random as npr
from scipy import interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from funkyyak import grad, numpy_wrapper as np
npr.seed(42)
from scipy.optimize import minimize
from redshift_utils import load_data_clean_split, project_to_bands, \
                           check_grad, fit_weights_given_basis, evaluate_random_direction
from slicesample import slicesample

## weighted loss function - observations have gaussian noise
def loss_fun(th_mat, X, inv_var):
    # unpack params
    N      = X.shape[0]
    omegas = th_mat[:, 0:N].T
    betas  = th_mat[:, N:]
    # compute like
    W = np.exp(omegas)
    B = np.exp(betas)
    B = B / np.sum(B, axis=1, keepdims=True)
    Xtilde = np.dot(np.exp(omegas), B)
    #Xtilde = reconstruct(th_mat)
    return np.sum( inv_var * np.square(X - Xtilde) )

def reconstruct(th_mat):
    omegas = th_mat[:, 0:N].T
    betas  = th_mat[:, N:]
    # compute like
    W = np.exp(omegas)
    B = np.exp(betas)
    B = B / np.sum(B, axis=1, keepdims=True)
    Xtilde = np.dot(np.exp(omegas), B)
    return Xtilde
loss_grad = grad(loss_fun)

## joint prior over parameters
def prior_loss(th_mat, N, sig2_omega=100., sig2_beta=10.):
    omegas = th_mat[:, 0:N].T
    betas  = th_mat[:, N:]
    return .5 / (sig2_omega) * np.sum(np.abs(omegas)) + \
           .5 / (sig2_beta) * np.sum(np.square(betas))
prior_loss_grad = grad(prior_loss)

## iteratively minimize weight/basis pairs
def train_iterative(th, X, Lam, max_iter=100): 

    def loss_omegas(omegas, B):
        ll_omega = 1 / (100.) * np.sum(np.square(omegas))
        Xtilde   = np.dot(np.exp(omegas), B)
        return np.sum( Lam * np.square(X - Xtilde) ) + ll_omega
    loss_omegas_grad = grad(loss_omegas)

    def loss_betas(betas, W):
        #ll_beta = np.sum(np.dot(np.dot(betas, KB_inv), betas.T))
        ll_beta = np.sum(np.square(betas))
        B = np.exp(betas)
        B = B / np.sum(B, axis = 1, keepdims=True)
        Xtilde = np.dot(W, np.exp(betas))
        return np.sum( Lam * np.square(X - Xtilde)) + ll_beta
    loss_betas_grad = grad(loss_betas)

    #### minimize w/ convex opt method
    print "    Iter       |    Train err  |   grad_mag  "
    N = X.shape[0]
    omegas = th[:, 0:N]
    betas  = th[:, N:]
    curr_loss = loss_fun(np.column_stack((omegas, betas)), X, Lam)
    lls = []
    for iter_i in range(max_iter):

        ## fix weights and minimize w.r.t basis
        W = np.exp(omegas).T
        res = minimize( x0 = betas.ravel(),
                        fun = lambda b: loss_betas(b.reshape(betas.shape), W),
                        jac = lambda b: loss_betas_grad(b.reshape(betas.shape), W).ravel(),
                        method = 'L-BFGS-B',
                        options = {'gtol':1e-6, 'disp':False, 'maxiter':30})
        betas = res.x.reshape(betas.shape)

        ## debug compute marignal loss
        print " post-basis opt loss: ", loss_fun(np.column_stack((omegas, betas)), X, Lam)

        ## now fix basis and minimize w.r.t. weights
        B = np.exp(betas)
        B = B / np.sum(B, axis=1, keepdims=True)
        res = minimize( x0 = omegas.ravel(),
                        fun = lambda o: loss_omegas(o.reshape(omegas.shape).T, B),
                        jac = lambda o: loss_omegas_grad(o.reshape(omegas.shape).T, B).ravel(),
                        method = 'L-BFGS-B',
                        options = {'gtol':1e-6, 'disp':False, 'maxiter':30})
        omegas = res.x.reshape(omegas.shape)

        ## compute marginal loss
        th = np.column_stack((omegas, betas))
        lls.append(loss_fun(th, X, Lam))
        th_grad = loss_grad(th, X, Lam)
        print "{0:15}|{1:15}|{2:15}".format(iter_i, "%5g"%lls[-1], np.sqrt(np.sum(th_grad*th_grad)))

    th_fine = np.column_stack((omegas, betas))
    return th_fine, lls


## simple gradient based NMF w/ gaussian noise training function
def train_model(th, X, Lam, cvx_iter=20000, sgd_iter = 1000,
                learning_rate = 1e-5, momentum = .9):
    print "    Iter       |    Train err  | Objective Fun |   step_size  "
    # Training parameters
    batch_size = 256

    # mix/match loss and regularizers 
    def target_grad(th): 
        lgrad = loss_grad(th, X, Lam)
        pgrad = prior_loss_grad(th, X.shape[0])
        return lgrad + pgrad

    cur_dir = np.zeros(th.shape)
    lls     = np.zeros(sgd_iter)
    for epoch in range(sgd_iter):
        grad_th    = target_grad(th)
        cur_dir    = momentum * cur_dir + (1.0 - momentum) * grad_th
        th        -= learning_rate * cur_dir
        lls[epoch] = loss_fun(th, X, Lam)

        step_mag = np.sqrt(np.sum(np.square(learning_rate*cur_dir)))
        if epoch % 100 == 0:
            print "{0:15}|{1:15}|{2:15}|{3:15}".format(epoch, 
                "%7g"%lls[epoch], 
                "%7g"%(lls[epoch] + prior_loss(th, X.shape[0])),
                "%2.4f"%step_mag)

    #### minimize w/ convex opt method
    print "Switching to CVX Optimization Method"
    th_shape = th.shape
    def lfun(th_vec): 
        return loss_fun(th_vec.reshape(th_shape), X, Lam) + \
               prior_loss(th_vec.reshape(th_shape), X.shape[0])
    def jfun(th_vec):
        return target_grad(th_vec.reshape(th_shape)).ravel()
    res = minimize(fun=lfun, jac=jfun, x0=th.ravel(), 
                    method = 'L-BFGS-B',
                    options = {'gtol':1e-6, 'disp':True, 'maxiter':cvx_iter})
    th_fine = np.reshape(res.x, th_shape)
    return th_fine, lls

if __name__=="__main__":

    ## load a handful of quasar spectra
    lam_obs, qtrain, qtest = \
        load_data_clean_split(spec_fits_file = 'quasar_data.fits', 
                              Ntrain = 400)

    ## first find a positive decomposition of quasar spectra on training data
    quasar_zerr    = qtrain['Z_err']
    N              = qtrain['spectra'].shape[0]

    ## initialize a basis using existing eigenQuasar Model
    header     = fitsio.read_header('../../data/eigen_specs/spEigenQSO-55732.fits')
    eigQSOfits = fitsio.FITS('../../data/eigen_specs/spEigenQSO-55732.fits')
    lam0       = 10.**(header['COEFF0'] + np.arange(header['NAXIS1']) * header['COEFF1'])
    lam0       = lam0[::5]
    eigQSO     = eigQSOfits[0].read()[:, ::5]
    K          = eigQSO.shape[0]

    ## resample to lam0 => rest frame basis 
    print "resampling de-redshifted data"
    wave_mat = np.zeros(qtrain['spectra'].shape)
    for i in range(qtrain['spectra'].shape[0]):
        wave_mat[i, :] = lam_obs / (1 + qtrain['Z'][i])
    spectra_resampled = np.zeros((qtrain['spectra'].shape[0], len(lam0)))
    spectra_ivar_resampled = np.zeros((qtrain['spectra'].shape[0], len(lam0)))
    #for i in range(qtrain['spectra'].shape[0]):
    #    if i%20==0: print "%d of %d"%(i, qtrain['spectra'].shape[0])
    #    spectra_resampled[i, :] = sinc_interp(wave_resampled,
    #                                        wave_mat[i, :],
    #                                        qtrain['spectra'][i, :],
    #                                        left = np.nan, right = np.nan)
    #    spectra_ivar_resampled[i, :] = sinc_interp(wave_resampled,
    #                                        wave_mat[i, :],
    #                                        quasar_ivar[i, :],
    #                                        left = np.nan, right = np.nan)
    for i in range(qtrain['spectra'].shape[0]):
        spectra_resampled[i, :] = np.interp(x     = lam0,
                                            xp    = wave_mat[i, :],
                                            fp    = qtrain['spectra'][i, :],
                                            left  = np.nan,
                                            right = np.nan)
        spectra_ivar_resampled[i, :] = np.interp(x     = lam0,
                                                 xp    = wave_mat[i, :],
                                                 fp    = qtrain['spectra_ivar'][i, :],
                                                 left  = np.nan,
                                                 right = np.nan)

    ## initial params proportional to Eigen QSO
    X                  = spectra_resampled
    X[np.isnan(X)]     = 0
    Lam                = spectra_ivar_resampled
    Lam[np.isnan(Lam)] = 0
    betas              = .001 * eigQSO.copy()
    betas             -= betas.max(axis=1, keepdims=True)
    omegas             = .001 * npr.randn(N, K)
    th                 = np.column_stack((omegas.T, betas))
    Nparam             = th.size

    ## or load from file
    th = np.load("cache/basis_th_K-%d_V-%d.npy"%B.shape)
    print loss_fun(th, X, Lam)

    ## sanity check gradient
    check_grad(fun = lambda th: loss_fun(th, X, Lam), 
               jac = lambda th: loss_grad(th, X, Lam),
               th  = th)

    ## train model
    th, lls = train_model(th, X, Lam, cvx_iter=10000, sgd_iter=500, 
                          learning_rate = 1e-5, momentum=.98)
    print "Fit loss: ", loss_fun(th, X, Lam)
    dth = loss_grad(th, X, Lam)
    print "Gradient mag: ", np.sqrt((dth*dth).sum())
    omegas = th[:, 0:N].T
    betas  = th[:, N:]
    W = np.exp(omegas).T
    B = np.exp(betas)
    B = B / B.sum(axis=1, keepdims=True)
    np.save("cache/basis_th_K-%d_V-%d.npy"%B.shape, th)
    np.save("cache/lls_K-%d_V-%d.npy"%B.shape, lls)
    np.save("cache/lam0_V-%d.npy"%len(lam0), lam0)

    ## sanity check out of sample fits
    if False:

        def plot_fits(qobj, idxs, refit_w = True, sgd_iter=1000):
            for n in idxs:
                fig   = plt.figure()
                z_n    = qobj['Z'][n]
                spec_n = qobj['spectra'][n, :]
                ivar_n = qobj['spectra_ivar'][n, :]
                if refit_w:
                    w = fit_weights_given_basis(B, lam0, spec_n, ivar_n, z_n, lam_obs, sgd_iter=sgd_iter)
                else:
                    w = W[:, n]
                plt.plot(lam_obs / (1 + z_n), spec_n)
                plt.plot(lam0, w.dot(B))
                plt.plot(lam_obs / (1 + z_n), np.sqrt(spec_ivar_n), color='grey', alpha=.5)
                plt.xlim( min(lam_obs/(1+z_n)), max(lam_obs/(1+z_n)) )
            plt.show()

        train_idxs = [20]
        plot_fits(qtrain, train_idxs, refit_w = False)
        plot_fits(qtrain, train_idxs, refit_w =True)

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
        test_idxs = [0, 10, 95, 140]
        plot_fits(qtest, test_idxs, refit_w = True)



    ## compute the likelihood redshift for one example
    #def z_likelihood(z, w, spec, spec_ivar, lam_obs): 
    #    # convert observation wavelengths to rest-frame wavelengths
    #    lam_rest = lam_obs / (1 + z)

    #    # interpolate observations to match basis
    #    spec_resampled = np.interp(x     = lam0,
    #                               xp    = lam_rest,
    #                               fp    = spec,
    #                               left  = np.nan,
    #                               right = np.nan)
    #    spec_ivar_resampled = np.interp(x     = lam0,
    #                                    xp    = lam_rest,
    #                                    fp    = spec_ivar,
    #                                    left  = np.nan,
    #                                    right = np.nan)
    #    spec_resampled[np.isnan(spec_resampled)] = 0
    #    spec_ivar_resampled[np.isnan(spec_ivar_resampled)] = 0

    #    # reconstruct model spectra
    #    spec_tilde = w.dot(b)
    #    ll = - spec_ivar_resampled.dot((spec_tilde - spec_resampled)**2)
    #    return ll

    #def prior_w(w): 
    #    if np.any(w <= 0): 
    #        return -np.inf
    #    return 0

    #n = 100
    #spec_n      = quasar_spectra[n, :]
    #spec_ivar_n = quasar_ivar[n, :]
    #w_n         = W[:, n]               # need to integrate this ouuuuut
    #print z_likelihood(quasar_z[n], w_n, spec_n, spec_ivar_n, lam_obs)
    #zs   = np.linspace(0, 5, 100)
    #llz  = np.array([z_likelihood(z, w_n, spec_n, spec_ivar_n, lam_obs) for z in zs])
    #plt.plot(zs, llz)
    #plt.vlines(x = quasar_z[n], ymin = llz.min(), ymax = llz.max(), label="measured redshift", linewidth=2)
    ##plt.vlines(x = quasar_z[n] - 2*quasar_zerr[n], ymin = llz.min(), ymax = llz.max())
    ##plt.vlines(x = quasar_z[n] + 2*quasar_zerr[n], ymin = llz.min(), ymax = llz.max())
    #plt.title("Red shift log likelihood")
    #plt.xlabel("$z$")
    #plt.ylabel("$\log p(z | obs)$")
    #plt.show()

    ### TODO: Slice sample this now integrate out uncertainty over w
    #Nsamps = 10000
    #ll_samps = np.zeros(Nsamps)
    #th_samps = np.zeros((Nsamps, len(w_n) + 1))
    #th_curr  = np.concatenate((w_n, [quasar_z[n]]))
    #lnpdf    = lambda th: z_likelihood(th[-1], th[:-1], spec_n, spec_ivar_n, lam_obs) + prior_w(th[:-1])
    #ll_curr  = lnpdf(th_curr)
    #for samp_i in range(Nsamps):
    #    th_curr, ll_curr = slicesample(th_curr, lnpdf, last_llh=ll_curr, 
    #                                   step=1, step_out=True,
    #                                   x_l = np.zeros(th_curr.shape), #everything is positive
    #                                   x_r = None,
    #                                   lb = -np.Inf, ub = np.Inf)
    #    th_samps[samp_i,:] = th_curr
    #    ll_samps[samp_i] = ll_curr
    #    if samp_i % 100 == 0:
    #        print samp_i

    #cnts, bins, patches = plt.hist(th_samps[(Nsamps/2):, -1], 100, alpha=.5, normed=True)
    #plt.xlabel("$z$ (red-shift)")
    #plt.ylabel("$p(z | X, B)$")
    #plt.vlines(quasar_z[n], 0,  cnts.max(), linewidth=2)
    #plt.vlines(quasar_z[n] - 2*quasar_zerr[n], 0, cnts.max(), linewidth=1)
    #plt.vlines(quasar_z[n] + 2*quasar_zerr[n], 0, cnts.max(), linewidth=1)
    #plt.show()


    

## run on quasar 
#def synthetic_example():
#    K = 3
#    P = 100
#    N = 100
#    kern = GPy.kern.rbf(input_dim = 1)
#    KB   = kern.K(np.linspace(0, 10, P).reshape((-1, 1))) + 1e-6 * np.eye(P)
#    KB_chol = np.linalg.cholesky(KB)
#    KB_inv = np.linalg.inv(KB)
#
#    def gen_params():
#        B = np.exp(np.linalg.cholesky(KB).dot(npr.randn(P, K)).T)
#        B /= B.sum(axis=1)[:, np.newaxis]
#        W = npr.rand(N, K)
#        sig2 = .01 * .01
#        X = W.dot(B) + np.sqrt(sig2)*npr.rand(N, P)
#
#        # test at location
#        omegas = np.log(W)
#        betas = np.log(B)
#        th = np.column_stack((omegas.T, betas))
#        Nparam = th.size
#        return B, W, th, X, Nparam, sig2
#    B, W, th, X, Nparam, sig2 = gen_params()
#    th0 = th.copy()
#
#    # Check grads
#    param_scale = .1
#    rand_dir = npr.randn(Nparam) * param_scale
#    rand_dir = rand_dir / np.sqrt(np.dot(rand_dir, rand_dir))
#    test_fun = lambda x : loss_fun(th + x * rand_dir.reshape(th.shape), X, sig2)
#    nd = (test_fun(1e-4) - test_fun(-1e-4)) / 2e-4
#    ad = np.dot(loss_grad(th, X, sig2).ravel(), rand_dir)
#    print "Checking grads. Relative diff is: {0}".format((nd - ad)/np.abs(nd))
#
#    # Train with sgd
#    #Binit, Winit, th, _, _, _ = gen_params()
#    #th = .1 * th
#    th, lls = train_model(th, X, sig2, max_iter = 10000)
#
#    print "TH0 loss: ", loss_fun(th0, X, sig2)
#    print "Fit loss: ", loss_fun(th, X, sig2)
#
#    Wfit = np.exp(th[:, 0:N]).T
#    Bfit = np.exp(th[:, N:])
#    Bfit = Bfit / Bfit.sum(axis=1)[:, np.newaxis] 
#
#    # compare basis
#    fig, axarr = plt.subplots(2, 1)
#    axarr[0].plot(B.T)
#    axarr[1].plot(Bfit.T)
#    plt.show()
#
#    # compare weights
#    fig, axarr = plt.subplots(1, 2)
#    axarr[0].imshow(Wfit[0:10, :], interpolation='none', vmin=Wfit.min(), vmax= Wfit.max())
#    axarr[1].imshow(W[0:10, :], interpolation='none', vmin=Wfit.min(), vmax=Wfit.max())
#    plt.show()
#
#    # 
#    fig = plt.figure()
#    Xtilde = Wfit.dot(Bfit)
#    plt.plot(Xtilde[0,:], label='$\hat x$')
#    plt.plot(X[0,:], label='$X$')
#    plt.title("Example reconstruction")
#    plt.legend()
#    plt.show()
#
