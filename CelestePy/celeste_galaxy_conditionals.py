"""
  Galaxy (single source) conditional distributions and gradients 
"""
import autograd.numpy as np
import CelestePy.mixture_profiles as mp
from autograd import grad
import CelestePy.util.like as like_util
import CelestePy.celeste_fast as celeste_fast
import scipy.stats
from util.bound.bounding_box import calc_bounding_radius
from util.like import gmm_like_2d

BANDS = ['u', 'g', 'r', 'i', 'z']
def galaxy_source_like(th, Z_s, images, check_overlap=True, unconstrained=True):
    """ log probability of Galaxy-specific pixels (z), conditioned on 
        galaxy parameters:
          - th     : [theta_s, sig_s, phi_s, rho_s]
          - u      : equatorial location
          - bs     : dict of fluxes (ugriz)
          - Z_s    : list of photon observations (list of 2d numpy arrays)
          - images : list of FitsImage objects (for equatorial => pixel and band info)
    """
    ll = 0.

    # unpack params
    theta_s, sig_s, phi_s, rho_s = th[0:4]
    R_s = gen_galaxy_transformation(sig_s, rho_s, phi_s)
    u_s = th[4:6]

    # compute rotation/scaling from params
    bs  = dict(zip(BANDS, th[-5:]))
    for n, img in enumerate(images):
        f_nms_exp   = gen_galaxy_prof_psf_image('exp', R_s, u_s, img)
        f_nms_dev   = gen_galaxy_prof_psf_image('dev', R_s, u_s, img)
        f_nms       = theta_s * f_nms_exp + (1. - theta_s) * f_nms_dev

        # convert source flux (nanomaggies) to image photon counts
        image_flux  = (bs[img.band] / img.calib) * img.kappa
        lam         = image_flux * f_nms
        ll         += np.sum(Z_s[n] * np.log(lam) - lam)
    return ll

def galaxy_source_like_grad(th, Z_s, images, check_overlap=True, unconstrained=False):
    # unpack shape params
    theta_s, sig_s, phi_s, rho_s = th[0:4]
    R_s = gen_galaxy_transformation(sig_s, rho_s, phi_s)

    # unpack location
    u_s = th[4:6]

    # unpack fluxes 
    bs = dict(zip(BANDS, th[-5:]))

    # generate profile-specific 
    grad_theta_s = 0.
    grad_bs      = dict(zip(BANDS, np.zeros(len(BANDS))))
    for n, img in enumerate(images):
        f_nms_exp  = gen_galaxy_prof_psf_image('exp', R_s, u_s, img)
        f_nms_dev  = gen_galaxy_prof_psf_image('dev', R_s, u_s, img)
        f_nms      = theta_s * f_nms_exp + (1. - theta_s) * f_nms_dev

        # compute gradient w.r.t theta_s
        f_nms_diff    = f_nms_exp - f_nms_dev
        grad_theta_s += np.sum((Z_s[n]/f_nms - bs[img.band])*f_nms_diff)

        # compute gradient w.r.t band fluxes
        grad_bs[img.band] += 1./bs[img.band]*np.sum(Z_s[n]) - np.sum(f_nms)

    # for parameters in the R matrix, just numerically differentiate
    numerical_inds = [1, 2, 3, 4, 5]
    grad_RU = np.zeros(5)
    for i,th_i in enumerate(numerical_inds):
        de       = np.zeros(th.shape)
        de[th_i] = 1e-5
        dell = galaxy_source_like(th+de, Z_s, images,
                                  check_overlap, unconstrained) - \
               galaxy_source_like(th-de, Z_s, images,
                                  check_overlap, unconstrained)
        grad_RU[i] = dell / (2.*de[th_i])

    # pack gradient
    th_grad = np.concatenate([
                [grad_theta_s],
                grad_RU,
                np.array([grad_bs[b] for b in BANDS])
              ])
    return th_grad

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


# galaxy profile objects - each is a mixture of gaussians
galaxy_profs = [mp.get_exp_mixture(), mp.get_dev_mixture()]
galaxy_prof_dict = dict(zip(['exp', 'dev'], galaxy_profs))

def gen_galaxy_prof_psf_image(prof_type, R, u, img, return_patch=True, xlim=None, ylim=None):
    """ generate the profile galaxy psf image given:
            - prof_type : either 'exp' or 'dev'
            - R_s       : the rotation of the ellipse (like a Cholesky
                          decomposition of a Covariance matrix)
            - u_s       : center of the profile
    """
    assert galaxy_prof_dict.has_key(prof_type), "unknown galaxy profile type"

    v_s = img.equa2pixel(u)

    # convolve image PSF and galaxy profile (generate mixture components)
    weights, means, covars = \
        celeste_fast.gen_galaxy_prof_psf_mixture_params(
           W             = np.dot(R, R.T),       #np.ndarray[FLOAT_t, ndim=2] W,
           v_s           = v_s,                  #np.ndarray[FLOAT_t, ndim=1] v_s,
           image_ws      = img.weights,         #np.ndarray[FLOAT_t, ndim=1] image_ws,
           image_means   = img.means,           #np.ndarray[FLOAT_t, ndim=2] image_means,
           image_covars  = img.covars,          #np.ndarray[FLOAT_t, ndim=3] image_covars,
           gal_prof_amp  = galaxy_prof_dict[prof_type].amp,       #np.ndarray[FLOAT_t, ndim=1] gal_prof_amp,
           gal_prof_sigs = galaxy_prof_dict[prof_type].var[:,0,0] #np.ndarray[FLOAT_t, ndim=1] gal_prof_sigs,
    )

    ERROR = 0.00001
    bound = calc_bounding_radius(weights, means, covars, ERROR, center=v_s)
    if xlim is not None and ylim is not None:
        minx_b, maxx_b = xlim
        miny_b, maxy_b = ylim
    else:
        minx_b, maxx_b = max(0, int(v_s[0] - bound)), min(int(v_s[0] + bound + 1), img.nelec.shape[1])
        miny_b, maxy_b = max(0, int(v_s[1] - bound)), min(int(v_s[1] + bound + 1), img.nelec.shape[0])
    y_grid = np.arange(miny_b, maxy_b, dtype=np.float)
    x_grid = np.arange(minx_b, maxx_b, dtype=np.float)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
    sub_pix_grid = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))

    psf_grid_small = gmm_like_2d(x       = sub_pix_grid,
                                 ws      = weights,
                                 mus     = means,
                                 sigs    = covars)

    if return_patch:
        return psf_grid_small.reshape(xx.shape, order='C'), (miny_b, maxy_b), (minx_b, maxx_b)

    # create full field grid
    psf_grid = np.zeros(img.nelec.shape)
    psf_grid[miny_b:maxy_b, minx_b:maxx_b] = \
        psf_grid_small.reshape(xx.shape, order='C')
    return psf_grid, (0, psf_grid.shape[0]), (0, psf_grid.shape[1])


def gen_galaxy_psf_image(th, u_s, img, xlim=None, ylim=None,
                                       check_overlap = True,
                                       unconstrained = True,
                                       return_patch=True):
    """ generates the profile of a combination of exp/dev images.
        Calls the above function twice - once for each profile, and adds them
        together
    """
    #unpack skew/location
    theta_s, sig_s, phi_s, rho_s = th[0:4]
    u_s       = np.array(u_s, dtype=np.float)
    R_s       = gen_galaxy_transformation(sig_s, rho_s, phi_s)
    f_nms_exp, ylime, xlime = \
        gen_galaxy_prof_psf_image('exp', R_s, u_s, img,
                                  return_patch=return_patch, xlim=xlim, ylim=ylim)
    f_nms_dev, ylimd, xlimd = \
        gen_galaxy_prof_psf_image('dev', R_s, u_s, img,
                                  return_patch=return_patch, xlim=xlim, ylim=ylim)
    #take the two patches above, align them, and return a single patch
    xlim = (np.min([xlime[0], xlimd[0]]), np.max([xlime[1], xlimd[1]]))
    ylim = (np.min([ylime[0], ylimd[0]]), np.max([ylime[1], ylimd[1]]))
    fpatch = np.zeros((ylim[1]-ylim[0], xlim[1]-xlim[0]))
    #exp contribution
    xe0 = xlime[0] - xlim[0]
    ye0 = ylime[0] - ylim[0]
    fpatch[ye0:(ye0 + ylime[1]-ylime[0]), xe0:(xe0 + xlime[1]-xlime[0])] += f_nms_exp * theta_s
    #dev contribution
    xd0 = xlimd[0] - xlim[0]
    yd0 = ylimd[0] - ylim[0]
    fpatch[yd0:(yd0 + ylimd[1]-ylimd[0]), xd0:(xd0 + xlimd[1]-xlimd[0])] += f_nms_dev * (1 - theta_s)
    return fpatch, ylim, xlim
    #f_nms     = theta_s * f_nms_exp + (1. - theta_s) * f_nms_dev
    #return f_nms, (0, f_nms.shape[0]), (0, f_nms.shape[1])


def gen_galaxy_psf_image_bound(src, img):
    """ generates the profile of a combination of exp/dev images.  
        Calls the above function twice - once for each profile, and adds them 
        together
    """
    #unpack skew/location
    u_s       = np.array(src.u, dtype=np.float)
    R_s       = gen_galaxy_transformation(src.sigma, src.rho, src.phi)
    Rexp      = gen_galaxy_prof_psf_image_bound('exp', R_s, u_s, img)
    Rdev      = gen_galaxy_prof_psf_image_bound('dev', R_s, u_s, img)
    return np.max([Rexp, Rdev])

def gen_galaxy_prof_psf_image_bound(prof_type, R, u, img, ERROR=.01):
    """given galaxy source params and an image psf, generate the bounding 
    radius that will encompass 1-epsilon of photons generated by source in 
    image """
    assert galaxy_prof_dict.has_key(prof_type), "unknown galaxy profile type"

    v_s = img.equa2pixel(u)

    # convolve image PSF and galaxy profile (generate mixture components)
    weights, means, covars = \
        celeste_fast.gen_galaxy_prof_psf_mixture_params(
           W             = np.dot(R, R.T),       #np.ndarray[FLOAT_t, ndim=2] W,
           v_s           = v_s,                  #np.ndarray[FLOAT_t, ndim=1] v_s,
           image_ws      = img.weights,         #np.ndarray[FLOAT_t, ndim=1] image_ws,
           image_means   = img.means,           #np.ndarray[FLOAT_t, ndim=2] image_means,
           image_covars  = img.covars,          #np.ndarray[FLOAT_t, ndim=3] image_covars,
           gal_prof_amp  = galaxy_prof_dict[prof_type].amp,       #np.ndarray[FLOAT_t, ndim=1] gal_prof_amp,
           gal_prof_sigs = galaxy_prof_dict[prof_type].var[:,0,0] #np.ndarray[FLOAT_t, ndim=1] gal_prof_sigs,
    )
    bound = calc_bounding_radius(weights, means, covars, ERROR, center=v_s)
    return bound


##########################################################################
# Drop-in prior probability functions
##########################################################################
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

##########################################################################
# Helper Methods
##########################################################################
#
#def det2d(K):
#    return K[0,0]*K[1,1] - K[1,0]*K[0,1]
#
#def inv2d(K):
#    return 1./det2d(K) * np.array([ [ K[1,1], -K[1,0] ],
#                                    [-K[0,1],  K[0,0] ] ])

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


if __name__=="__main__":

    # test the gradients for some random parameter setting
    import sys
    from glob import glob
    narg     = len(sys.argv)
    data_dir = str(sys.argv[1]) if narg > 1 else '/Users/acm/Dropbox/Proj/astro/DESIMCMC/data/experiment_stamps/'

    # Grab images, catalog data and initialize galaxy source
    from CelestePy.util.misc import init_utils, check_grad
    cat_glob = glob(data_dir + '/cat*.fits')[0:1]
    cat_srcs, imgs, teff_catalog, us = init_utils.load_imgs_and_catalog(cat_glob)

    ## create srcs images
    init_draw = lambda: (np.random.beta(.4, .4) + .01) / 1.02
    srcs = init_utils.init_sources_from_image_block(imgs[0:5])[0:1]
    srcs[0]        = init_utils.init_random_galaxy(srcs[0].u)
    srcs[0].phi    = init_draw() * np.pi
    srcs[0].sigma  = 10*np.random.rand()
    srcs[0].rho    = np.random.beta(.4, .4)
    srcs[0].theta  = np.random.beta(.4, .4)
    srcs[0].fluxes = dict(zip(BANDS, 20*np.random.rand(len(BANDS))))

    # make sure likelihood evaluates
    th = np.concatenate((
        [srcs[0].theta, srcs[0].sigma, srcs[0].phi, srcs[0].rho],
        srcs[0].u,
        [srcs[0].fluxes[b] for b in BANDS]))
    Z_s = [img.nelec for img in imgs]
    ll = galaxy_source_like(th, Z_s, imgs, check_overlap=True, unconstrained=True)
    ll_grad = galaxy_source_like_grad(th, Z_s, imgs)

    print ll
    print ll_grad

    # check to make sure the gradient is correct
    check_grad(fun = lambda(th): galaxy_source_like(th, Z_s, imgs),
               jac = lambda(th): galaxy_source_like_grad(th, Z_s, imgs),
               th  = th, compwise=True)


    # do quick gradient ascent
    from scipy.optimize import minimize
    res = minimize(fun = lambda(th): -galaxy_source_like(th, Z_s, imgs),
                   jac = lambda(th): -galaxy_source_like_grad(th, Z_s, imgs),
                   x0  = res.x,
                   method = 'L-BFGS-B')

    print galaxy_source_like(res.x, Z_s, imgs)
    check_grad(fun = lambda(th): galaxy_source_like(th, Z_s, imgs),
               jac = lambda(th): galaxy_source_like_grad(th, Z_s, imgs),
               th  = res.x, compwise=True)

#################
# Dead code
##################

#def gen_galaxy_psf_mixture_components_fast(v_s, W, thetas, image_ws,
#                                           image_means, image_covars):
#    """ wrapper for the Cython version of the above function """
#    return celeste_fast.gen_galaxy_psf_mixture_params(
#        thetas = thetas,                     #np.ndarray[FLOAT_t, ndim=1] thetas,
#        W      = W,                          #np.ndarray[FLOAT_t, ndim=2] W,
#        v_s    = v_s,                        #np.ndarray[FLOAT_t, ndim=1] v_s,
#        image_ws = image_ws,                 #np.ndarray[FLOAT_t, ndim=1] image_ws,
#        image_means = image_means,           #np.ndarray[FLOAT_t, ndim=2] image_means,
#        image_covars = image_covars,          #np.ndarray[FLOAT_t, ndim=3] image_covars,
#        gal_exp_amp = galaxy_profs[0].amp,   #np.ndarray[FLOAT_t, ndim=1] gal_exp_amp,
#        gal_exp_sigs = galaxy_profs[0].var[:,0,0],  #np.ndarray[FLOAT_t, ndim=1] gal_exp_sigs,
#        gal_dev_amp  = galaxy_profs[1].amp,  #np.ndarray[FLOAT_t, ndim=1] gal_dev_amp,
#        gal_dev_sigs = galaxy_profs[1].var[:,0,0])  #np.ndarray[FLOAT_t, ndim=1] gal_dev_sigs
#
#def gen_galaxy_psf_mixture_components(v_s, W, thetas, image_ws, image_means, image_covars):
#    """ computes mixture components of a PSF convolved with a mixture of 
#        galaxy types
#    """
#    num_components = len(image_ws) * sum([len(gp.amp) for gp in galaxy_profs])
#    weights = np.zeros(num_components) 
#    means   = np.zeros((num_components, 2)) 
#    covars  = np.zeros((num_components, 2, 2))
#    cnt     = 0
#    for k in range(len(image_ws)):                 # num PSF Componenets
#        for i in range(2):                              # two galaxy types
#            for j in range(len(galaxy_profs[i].amp)):   # galaxy type components
#                weights[cnt] = image_ws[k] * thetas[i] * galaxy_profs[i].amp[j]
#                means[cnt,:] = v_s + image_means[k,:]
#                covars[cnt, :, :] = image_covars[k,:,:] + \
#                                    np.dot(galaxy_profs[i].var[j,:,:], W)
#                cnt += 1
#    return weights, means, covars

#def galaxy_skew_like(th, u, fluxes, Z_s, images,
#                     check_overlap = True,
#                     pixel_grid    = None,
#                     unconstrained = True):
#    """ log probability of Galaxy-specific pixels (z), conditioned on 
#        galaxy parameters:
#          - th     : [theta_s, sig_s, phi_s, rho_s]
#          - u      : equatorial location
#          - bs     : dict of fluxes (ugriz)
#          - Z_s    : list of photon observations (list of 2d numpy arrays)
#          - images : list of FitsImage objects (for equatorial => pixel and band info)
#    """
#    ll = 0.
#    for n, img in enumerate(images):
#        gal_prof_psf = gen_galaxy_psf_image(th[0:4], u, img,
#                                            pixel_grid    = pixel_grid,
#                                            unconstrained = unconstrained)
#        # convert source flux (nanomaggies) to image photon counts
#        image_flux   = (fluxes[img.band] / img.calib) * img.kappa
#        lam          = image_flux * gal_prof_psf
#        ll          += np.sum(Z_s[n] * np.log(lam) - lam)
#    return ll
#galaxy_skew_like_grad = grad(galaxy_skew_like)


