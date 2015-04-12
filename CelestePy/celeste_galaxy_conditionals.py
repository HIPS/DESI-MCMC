"""
  Galaxy (single source) conditional distributions and gradients 
"""
import numpy as np
import CelestePy.mixture_profiles as mp
from autograd import grad
from CelestePy.util.like import fast_inv_gamma_lnpdf
from CelestePy.util.like.gmm_like_fast import gmm_like_2d
import CelestePy.celeste_fast as celeste_fast
#from CelestePy.util.like import ein_gmm_like as gmm_like

def galaxy_source_like(th, Z_s, images, check_overlap=True, pixel_grid=None, unconstrained=True):
    """ log probability of Galaxy-specific pixels (z), conditioned on 
        galaxy parameters:
          - th     : [theta_s, sig_s, phi_s, rho_s]
          - u      : equatorial location
          - bs     : dict of fluxes (ugriz)
          - Z_s    : list of photon observations (list of 2d numpy arrays)
          - images : list of FitsImage objects (for equatorial => pixel and band info)
    """
    ll = 0.
    bs = dict(zip(['u', 'g', 'r', 'i', 'z'], th[-5:]))
    for n, img in enumerate(images):
        gal_prof_psf = gen_galaxy_psf_image(th[0:4], th[4:6], img, pixel_grid=pixel_grid, unconstrained=unconstrained)
        # convert source flux (nanomaggies) to image photon counts
        image_flux   = (bs[img.band] / img.calib) * img.kappa
        lam          = image_flux * gal_prof_psf
        ll          += np.sum(Z_s[n] * np.log(lam) - lam)
    return ll + galaxy_prior(th[0], th[1], th[2], th[3])
galaxy_source_like_grad = grad(galaxy_source_like)


def galaxy_skew_like(th, u, fluxes, Z_s, images,
                     check_overlap = True,
                     pixel_grid    = None,
                     unconstrained = True):
    """ log probability of Galaxy-specific pixels (z), conditioned on 
        galaxy parameters:
          - th     : [theta_s, sig_s, phi_s, rho_s]
          - u      : equatorial location
          - bs     : dict of fluxes (ugriz)
          - Z_s    : list of photon observations (list of 2d numpy arrays)
          - images : list of FitsImage objects (for equatorial => pixel and band info)
    """
    ll = 0.
    for n, img in enumerate(images):
        gal_prof_psf = gen_galaxy_psf_image(th[0:4], u, img,
                                            pixel_grid    = pixel_grid,
                                            unconstrained = unconstrained)
        # convert source flux (nanomaggies) to image photon counts
        image_flux   = (fluxes[img.band] / img.calib) * img.kappa
        lam          = image_flux * gal_prof_psf
        ll          += np.sum(Z_s[n] * np.log(lam) - lam)
    return ll
galaxy_skew_like_grad = grad(galaxy_skew_like)

def gen_galaxy_transformation(sig_s, rho_s, phi_s):
    """ from dustin email, Jan 27
        sig_s (re)  : arcsec (greater than 0)
        rho_s (ab)  : axis ratio, dimensionless, in [0,1]
        phi_s (phi) : radians, "E of N", 0=direction of increasing Dec,
                      90=direction of increasing RAab = 
    """
    # convert re, ab, phi into a transformation matrix
    # convert unit vector to degrees
    re_deg = max(1./30, sig_s) / 3600.
    cp     = np.cos(phi_s)
    sp     = np.sin(phi_s)

    # Squish, rotate, and scale into degrees.
    # resulting G takes unit vectors (in r_e) to degrees
    # (~intermediate world coords)
    G = re_deg * np.array([[ cp, sp * rho_s], 
                           [-sp, cp * rho_s]])

    # "cd" takes pixels to degrees (intermediate world coords)
    cd = np.array([[0.396/3600, 0.         ],
                   [0.,         0.396/3600.]])

    # T takes pixels to unit vectors (effective radii).
    T    = np.dot(np.linalg.inv(G), cd)
    Tinv = np.linalg.inv(T)
    return Tinv



galaxy_profs = [mp.get_exp_mixture(), mp.get_dev_mixture()]
def gen_galaxy_psf_mixture_components(v_s, W, thetas, image_ws, image_means, image_covars):
    """ computes mixture components of a PSF convolved with a mixture of 
        galaxy types
    """
    num_components = len(image_ws) * sum([len(gp.amp) for gp in galaxy_profs])
    weights = np.zeros(num_components) 
    means   = np.zeros((num_components, 2)) 
    covars  = np.zeros((num_components, 2, 2))
    cnt     = 0
    for k in range(len(image_ws)):                 # num PSF Componenets
        for i in range(2):                              # two galaxy types
            for j in range(len(galaxy_profs[i].amp)):   # galaxy type components
                weights[cnt] = image_ws[k] * thetas[i] * galaxy_profs[i].amp[j]
                means[cnt,:] = v_s + image_means[k,:]
                covars[cnt, :, :] = image_covars[k,:,:] + \
                                    np.dot(galaxy_profs[i].var[j,:,:], W)
                cnt += 1
    return weights, means, covars


def gen_galaxy_psf_mixture_components_fast(v_s, W, thetas, image_ws, image_means, image_covars):
    """ wrapper for the Cython version of the above function """
    return celeste_fast.gen_galaxy_psf_mixture_params(
        thetas = thetas,                     #np.ndarray[FLOAT_t, ndim=1] thetas,
        W      = W,                          #np.ndarray[FLOAT_t, ndim=2] W,
        v_s    = v_s,                        #np.ndarray[FLOAT_t, ndim=1] v_s,
        image_ws = image_ws,                 #np.ndarray[FLOAT_t, ndim=1] image_ws,
        image_means = image_means,           #np.ndarray[FLOAT_t, ndim=2] image_means,
        image_covars = image_covars,          #np.ndarray[FLOAT_t, ndim=3] image_covars,
        gal_exp_amp = galaxy_profs[0].amp,   #np.ndarray[FLOAT_t, ndim=1] gal_exp_amp,
        gal_exp_sigs = galaxy_profs[0].var[:,0,0],  #np.ndarray[FLOAT_t, ndim=1] gal_exp_sigs,
        gal_dev_amp  = galaxy_profs[1].amp,  #np.ndarray[FLOAT_t, ndim=1] gal_dev_amp,
        gal_dev_sigs = galaxy_profs[1].var[:,0,0])  #np.ndarray[FLOAT_t, ndim=1] gal_dev_sigs

def gen_galaxy_psf_image(th, u_s, image,
                         check_overlap = True,
                         pixel_grid    = None,
                         unconstrained = True):
    """ Q function conditioned on galaxies, pass in galaxy-
        specific parameters:
        th = [theta_s, sig_s, phi_s, rho_s, b_su, b_sg, ..., b_sz]
    """
    if unconstrained:
        theta_s, sig_s, phi_s, rho_s = constrain_params(th)
    else:
        theta_s, sig_s, phi_s, rho_s = th
    v_s = image.equa2pixel(u_s)

    # compute rotation/scaling from params
    R = gen_galaxy_transformation(sig_s, rho_s, phi_s)
    W = np.dot(R, R.T)

    # compute MOG components
    thetas = np.array([theta_s, 1. - theta_s])
    weights, means, covars = \
        gen_galaxy_psf_mixture_components_fast(
            v_s, W,
            thetas       = np.array([theta_s, 1. - theta_s]),
            image_ws     = image.weights,
            image_means  = image.means,
            image_covars = image.covars)

    # weights0, means0, covars0 = \
    #     gen_galaxy_psf_mixture_components(v_s, W, thetas, 
    #                                       image_ws    =image.weights, 
    #                                       image_means = image.means, 
    #                                       image_covars = image.covars)

    # instantiate a pixel grid if necessary
    if pixel_grid is None: 
        y_grid = np.arange(image.nelec.shape[0], dtype=np.float) + 1
        x_grid = np.arange(image.nelec.shape[1], dtype=np.float) + 1
        yy, xx = np.meshgrid(x_grid, y_grid, indexing='xy')
        pixel_grid = np.column_stack((xx.ravel(), yy.ravel()))

    ## evaluate equation 11-13 in jeff's november writeup
    psf_grid = fast_gmm_like(x = pixel_grid, 
                             ws = weights,
                             mus = means,
                             sigs = covars)
    return psf_grid.reshape(image.nelec.shape).T

def fast_gmm_like(x, ws, mus, sigs): 
    N_elem = np.atleast_1d(x).shape[0]
    probs = np.zeros(N_elem)
    gmm_like_2d(probs, x, np.array(ws), np.array(mus), np.array(sigs))
    return probs

def det2d(K):
    return K[0,0]*K[1,1] - K[1,0]*K[0,1]

def inv2d(K):
    return 1./det2d(K) * np.array([ [ K[1,1], -K[1,0] ],
                                    [-K[0,1],  K[0,0] ] ])

def galaxy_shape_prior_unconstrained(logit_theta, log_sig, logit_phi, logit_rho): 
    return -(1./50.) * (logit_theta * logit_theta + \
                        log_sig     * log_sig + \
                        logit_phi   * logit_phi + \
                        logit_rho   * logit_rho)

def galaxy_shape_prior_constrained(theta, sig, phi, rho):
    #confirm correct ranges
    if theta <= 0. or theta >= 1. or \
            sig <= 0. or \
            phi <= 0. or phi >= np.pi or \
            rho <= 0. or rho >= 1.:
        return -np.inf
    return fast_inv_gamma_lnpdf(sig*sig, a0=1., b0=1.)

def constrain_params(th):
    """ takes unconstrained parameters, and constrains them to 
        th = [theta_s, sig_s, phi_s, rho_s]
          theta_s => [0, 1]
          sig_s   => [0, \infty]
          phi_s   => [0, pi]
          rho_s   => [0, 1]
    """
    return  1./(1.+np.exp( -th[0] )), \
            np.exp( th[1] ), \
            np.pi / (1. + np.exp(-th[2])), \
            1./(1.+np.exp(-th[3]))

def unconstrain_params(th_constrained):
    """ takes constrained parameters and sets them free.
        th = [theta_s, sig_s, phi_s, rho_s]
          theta_s => [0, 1]
          sig_s   => [0, \infty]
          phi_s   => [0, pi]
          rho_s   => [0, 1]
    """
    return np.log(th_constrained[0] / (1. - th_constrained[0])), \
           np.log(th_constrained[1]), \
           np.log(th_constrained[2] / (np.pi - th_constrained[2])), \
           np.log(th_constrained[3] / (1. - th_constrained[3]))



