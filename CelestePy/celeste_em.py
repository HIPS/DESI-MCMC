import CelestePy.planck
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.optimize import fmin 
from CelestePy import SrcParams
from CelestePy import celeste_likelihood_multi_image, \
                      gen_src_prob_layers, \
                      gen_point_source_psf_image, \
                      gen_src_image
from CelestePy.util.misc.plot_util import compare_to_model, subplot_imshow_colorbar
from CelestePy.util.infer.slicesample import slicesample
from CelestePy.util.misc.timer_util import *
from CelestePy.celeste_galaxy_conditionals import galaxy_source_like, galaxy_source_like_grad
import CelestePy.mixture_profiles as mp

def celeste_em(srcs, imgs, maxiter=20, debug=False, verbose=True): 
    """ maximizes log likelihood over fixed-num-source parameters 
        Input: 
            srcs: python list of SrcParams
            imgs: python list of FitsImage objects
            maxiter : max number of EM iterations
            debug : turn on plotting
    """
    prev_ll = celeste_likelihood_multi_image(srcs, imgs)
    ll_trace = [prev_ll]

    # cache the unique bands present in the dataset - if it's a small number 
    # we can cut a few corners computing the number of expected photons per 
    # joule in each band for a given temperature
    imgbands    = np.array([img.band for img in imgs])
    uniquebands = np.unique(imgbands)

    printif("Initial Log Likelihood = %2.2f"%prev_ll, verbose)
    for em_iter in range(maxiter):
        printif("============================================", verbose)

        ## E-step, generate sub-images 
        printif("  iter %d E-step"%em_iter, verbose)
        all_src_probs  = []
        for img in imgs: 
            src_probs = gen_src_prob_layers(srcs, img)

            #### debug #####
            if debug:
                if img.band == 'r':
                    #compare_to_model(srcs, img)
                    fig, axarr = plt.subplots(1, src_probs.shape[0])
                    for s in range(src_probs.shape[0]):
                        im = axarr[s].imshow(img.nelec*src_probs[s,:,:], interpolation='none', origin='lower') 
                        fig.colorbar(im)
                    plt.show()
            #### end debug ####

            # save for M step
            all_src_probs.append(src_probs)

        ## M-step
        printif("  iter %d M-step"%em_iter, verbose)

        # compute optimal noise param for each image
        for i,img in enumerate(imgs):
            eps_tmp = img.epsilon
            img.epsilon = np.sum(img.nelec * src_probs[0,:,:]) / (img.nelec.size)
            printif("      img %d eps %2.2f => %2.2f (eps0 = %2.2f)"%(i, eps_tmp, img.epsilon, img.epsilon0), verbose > 1)

        # compute optimal params for each source
        for s in range(len(srcs)):

            #### joint Temp, Brightness source specific opt func (Q function), 
            #### unused - the profile below works
            #def src_loss(th):
            #    ll = 0
            #    for i, img in enumerate(imgs): 
            #        expected_num_photons = planck.photons_expected_brightness(th[0], th[1], img.band)
            #        ll += np.sum(img.nelec * all_src_probs[i][s+1,:,:]) * np.log(expected_num_photons) - expected_num_photons
            #    return -ll

            # helper maximization functions
            def compute_image_statistics(temp): 
                X_tildes = np.zeros(len(imgs))  # num photons source s is reponsible for in image n (p_n,m,s * x_n,m)
                # fraction of photons you'd expect to see due to point spread
                # function - if the entire source is in the image, this should 
                # one, if the source is far away from this image, this should be 0
                sum_fs   = np.zeros(len(imgs))  
                for n, img in enumerate(imgs):
                    # compute sum x_{n,m} p_{n,m,s}
                    X_tildes[n] = np.sum(all_src_probs[n][s+1,:,:] * img.nelec)

                    # compute fraction of photons this image will see
                    sum_fs[n] = min(1., np.sum(gen_point_source_psf_image(srcs[s].u, img)))
                return X_tildes, sum_fs

            # instead of re-computing for each image band, we can just cache 
            # the result and reference it later.  All 'r' band images yield 
            # the same 'photon per joule' count at temperature t
            # WOOO this is about 10x faster!
            def compute_photons_per_joule_per_image(t, imgbands): 
                I_ts = np.zeros(len(imgbands))
                for i, b in enumerate(uniquebands):
                    I_ts[imgbands==b] = planck.photons_per_joule(t, b)

                ######## DEBUG ###############################################
                if False:
                    I_ts_slow = np.array([planck.photons_per_joule(temp, img.band) for img in imgs])
                    assert np.all(I_ts_slow == I_ts), 'photons per joule per image do not match!'
                ###############################################################
                return I_ts

            # Cache image statistics that are constant - the PSF computation is 
            # expensive and not necessary at this point
            X_tildes, sum_fs = compute_image_statistics(srcs[s].t)

            # profile likelihood - only of temperature
            def partial_loss(temp): 
                # compute I(t_s, beta_n) - num photons you'd expect to see
                I_ts = compute_photons_per_joule_per_image(temp, imgbands)
                return X_tildes.dot(np.log(I_ts)) - \
                       np.log(I_ts.dot(sum_fs)) * X_tildes.sum()

            ######### DEBUG ##################################################
            if debug:
                tgrid = np.linspace(500, 10000, 100)
                bgrid = np.linspace(1e-16, 1e-10, 100)
                tt, bb = np.meshgrid(tgrid, bgrid)
                zz = np.array([src_loss(th) for th in np.column_stack((tt.ravel(), bb.ravel()))])
                plt.contour(tgrid, bgrid, zz.reshape(tt.shape))
                #ellt  = np.array([partial_loss(t) for t in tgrid])
                #plt.plot(tgrid, ellt)
                plt.show()
            ######## END DEBUG ###############################################

            ## 1) maximize t_s partial likelihood (one dimensional problem)
            t_hat = fmin(lambda(t): -partial_loss(t), srcs[s].t, disp=False)
            t_hat = t_hat[0]

            ## 2) compute maximal brightness \hat b_s( \hat t_s )
            #I_ts = np.array([planck.photons_per_joule(t_hat, img.band) for img in imgs])
            I_ts  = compute_photons_per_joule_per_image(t_hat, imgbands)
            fac   = 1./(planck.lens_area * planck.exposure_duration * \
                        planck.sun_wattage / (planck.m_per_ly**2))
            b_hat = fac * (1./I_ts.dot(sum_fs)) * X_tildes.sum()
            printif("   src %d temp       = %2.2f => %2.2f"%(s, srcs[s].t, t_hat), verbose>1)
            printif("   src %d brightness = %2.2f => %2.2f"%(s, srcs[s].b, b_hat), verbose>1)
            srcs[s].t = t_hat
            srcs[s].b = b_hat

            ## TODO - maximize u_s (this will probably have to be incorporated 
            ## into a joint update of t_s (or an iterative routine)
            ## 3) maximize u_s given \hat b_s, \hat t_s


            ##############################
            ## Galaxy params optimization
            ## 
            #galaxy_source_like(th, u, bs, Z_s, images, check_overlap=True, pixel_grid=None):
 

        ll = celeste_likelihood_multi_image(srcs, imgs)
        ll_trace.append(ll)
        printif(".... current marginal likelihood = %2.2f"%ll, verbose)
        if prev_ll > ll: 
            printif("marginal likelihood DECREASED!!!", verbose)
            printif("   %2.4f => %2.4f"%(prev_ll, ll), verbose)
            printif("   probably a bug or a failure of the internal maximization scheme", verbose)
        if ll - prev_ll < 1: 
            print "marginal likelihood converging, stopping (iter = %d, maxiter = %d)"%(em_iter, maxiter)
            break
        prev_ll = ll
    return ll_trace, em_iter < maxiter




def unif_lnpdf(x, a0, b0):
    if x <= a0 or x >= b0:
      return -np.inf
    return 0.

def fast_gamma_lnpdf(x, a0, b0): 
    """ Unnormalized gamma log pdf.  a0 = shape, b0 = rate """
    if x <= 0:
        return -np.inf
    return (a0-1.)*np.log(x) - b0*x


# priors over galaxy parameters
half_pi = np.pi/2
def galaxy_log_prior(theta, phi, sigma, rho):
    # uniform between devac and expo
    if theta <= 0. or theta >= 1.:
        return -np.inf
    # uniform between [0, pi/2]
    if phi <= 0. or phi >= half_pi:
        return -np.inf
    # ratio of minor/major axis length is bewteen 0,1
    if rho <=0. or rho >= 1.:
        return -np.inf
    # scale free prior 
    return - np.log(sigma)

def printif(statement, condition):
    if condition:
        print statement

if __name__=="__main__":
    pass


