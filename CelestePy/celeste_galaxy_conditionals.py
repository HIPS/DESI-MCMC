"""
Galaxy (single source) conditional distributions and gradients 
"""
import numpy as np
import CelestePy.mixture_profiles as mp
from autograd import grad
from CelestePy.util.like import fast_inv_gamma_lnpdf

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
    thetas = [theta_s, 1. - theta_s]
    num_components = len(image.weights) * sum([len(gp.amp) for gp in galaxy_profs])
    weights = []
    means   = []
    covars  = []
    for k in range(len(image.weights)):                 # num PSF Componenets
        for i in range(2):                              # two galaxy types
            for j in range(len(galaxy_profs[i].amp)):   # galaxy type components
                imgw = image.weights[k] * thetas[i] * galaxy_profs[i].amp[j]
                weights.append(imgw)
                means.append(v_s + image.means[k,:])
                covars.append(image.covars[k,:,:] + \
                              np.dot(galaxy_profs[i].var[j,:,:], W))

    # instantiate a pixel grid if necessary
    if pixel_grid is None: 
        y_grid = np.arange(image.nelec.shape[0], dtype=np.float) + 1
        x_grid = np.arange(image.nelec.shape[1], dtype=np.float) + 1
        yy, xx = np.meshgrid(x_grid, y_grid, indexing='xy')
        pixel_grid = np.column_stack((xx.ravel(), yy.ravel()))

    ## evaluate equation 11-13 in jeff's november writeup
    psf_grid = gmm_like(x = pixel_grid, 
                        ws = weights,
                        mus = means,
                        sigs = covars)
    return psf_grid.reshape(image.nelec.shape).T

Z = 0.15915494309189535
def gmm_like(x, ws, mus, sigs):
    N_elem = np.atleast_1d(x).shape[0] # number of rows of data
    probs = np.zeros(N_elem)
    for k in range(len(ws)):
        inv_detK  = 1. / np.sqrt(np.linalg.det(sigs[k]))
        K_inv     = np.linalg.inv(sigs[k])
        quad_term = np.sum(np.dot(x-mus[k], K_inv) * (x - mus[k]), axis=1, keepdims=False)
        probs     = probs + ws[k] * Z * inv_detK * np.exp(-.5 * quad_term)
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



