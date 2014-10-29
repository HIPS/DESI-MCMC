import planck
import numpy as np
from scipy.stats import gamma
from scipy.optimize import fmin 
from celeste import celeste_likelihood_multi_image, gen_src_prob_layers, gen_point_source_psf_image, gen_src_image
from util.plot_util import compare_to_model, subplot_imshow_colorbar
from mcmc_transitions import sampleAuxSourceCounts
from util.slicesample import slicesample
import matplotlib.pyplot as plt
from util.timer_util import *

def celeste_em(srcs, imgs, maxiter=20, debug=False, verbose=True): 
    """ maximizes log likelihood over fixed-num-source parameters 
        Input: 
            srcs: python list of PointSrcParams
            imgs: python list of FitsImage objects
            maxiter : max number of EM iterations
            debug : turn on plotting
    """
    prev_ll = celeste_likelihood_multi_image(srcs, imgs)
    ll_trace = [prev_ll]

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

            # Cache image statistics that are constant - the PSF computation is 
            # expensive and not necessary at this point
            X_tildes, sum_fs = compute_image_statistics(srcs[s].t)

            # profile likelihood - only of temperature
            def partial_loss(temp): 
                # compute I(t_s, beta_n) - num photons you'd expect to see
                I_ts = np.array([planck.photons_per_joule(temp, img.band) for img in imgs])
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
            I_ts = np.array([planck.photons_per_joule(t_hat, img.band) for img in imgs])
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



def celeste_gibbs_sample(srcs, imgs, subiter=2, debug=False, verbose=True): 
    """ Does a single gibbs sweep - samples all Z_{n,m,s} and then samples
        all u_s, t_s, b_s for each source (repeats subiter times)
        Input: 
            srcs: python list of PointSrcParams
            imgs: python list of FitsImage objects
            maxiter : max number of EM iterations
            debug : turn on plotting
    """

    # for each image, sample the source specific counts (Z_{n,m,s})
    printif("    sampling Z's", verbose)
    all_src_images = []
    for img in imgs:
        #src_image = sampleAuxSourceCounts(srcs, img, eta=0)
        src_probs = gen_src_prob_layers(srcs, img)
        src_image = np.zeros(src_probs.shape)
        for (i,j), xij in np.ndenumerate(img.nelec):
            src_image[:,i,j] = np.random.multinomial(int(xij), src_probs[:,i,j])
        all_src_images.append(src_image)

        #### debug #####
        if debug:
            if img.band == 'r':
                fig, axarr = plt.subplots(1, src_image.shape[0])
                subplot_imshow_colorbar(src_image, fig, axarr)
                plt.show()
        #### end debug ####
    printif("      .... done", verbose)

    # compute optimal noise param for each image
    printif("    sampling image specific epsilons", verbose)
    a_0 = 5      # convolution parameter - higher tends to avoid 0
    b_0 = .005   # inverse scale parameter
    for i,img in enumerate(imgs):
        a_n = a_0 + np.sum(all_src_images[i][0,:,:])
        b_n = b_0 + img.nelec.size
        eps_tmp = img.epsilon
        img.epsilon = np.random.gamma(a_n, 1./b_n)
        printif("      img %d eps %2.2f => %2.2f (eps0 = %2.2f)"%(i, eps_tmp, img.epsilon, img.epsilon0), 
                verbose and i < 5)
    printif("      .... done", verbose)

    # for each source, sample source specific params
    # TODO: add some sort of prior over brightness and temp (maybe location)
    for s in range(len(srcs)):

        #### joint Temp, Brightness source specific opt func
        def temp_bright_like(th, fs_sum):
            ll = 0
            for i, img in enumerate(imgs): 
                expected_num_photons = fs_sum[i]*planck.photons_expected_brightness(th[0], th[1], img.band)
                if expected_num_photons > 0:
                    ll += np.sum(all_src_images[i][s+1,:,:]) * np.log(expected_num_photons) - expected_num_photons
            ll += gamma(2, scale=1/.0002).logpdf(th[0])
            ll += gamma(1., scale=1.).logpdf(th[1])
            return ll

        #### likelihood factor that only depends on locatio
        def loc_like(u): 
            ll = 0
            srcs[s].u = u
            for n, img in enumerate(imgs):
                f_sn = gen_src_image(srcs[s], img)

                # mask to remove the negative infinities
                mask = all_src_images[n][s+1,:,:] > 0
                ll  += np.sum(np.log(f_sn[mask])*all_src_images[n][s+1,mask]) \
                       - f_sn.sum()
            return ll

        #### iterate a bunch - sample locs and brightness/temps
        for gibbs_iter in range(subiter):

            # sample temp/brightness
            printif( "    source %d: slice sampling t's and b's"%s, verbose)

            # cache existing state
            tmp_t = srcs[s].t
            tmp_b = srcs[s].b

            # compute fraction of photons this image will see
            sum_fs   = np.zeros(len(imgs))  
            for n, img in enumerate(imgs):
                sum_fs[n] = min(1., np.sum(gen_point_source_psf_image(srcs[s].u, img)))
            for it in range(2):
                th     = np.array([srcs[s].t, srcs[s].b])
                th, ll = slicesample(xx       = th,
                                     llh_func = lambda(th): temp_bright_like(th, sum_fs),
                                     step     = [1000, .1], 
                                     step_out = True,
                                     x_l      = [0., 0.])
            srcs[s].t = th[0]
            srcs[s].b = th[1]
            printif("        t: %2.2f => %2.2f"%(tmp_t, srcs[s].t),
                    verbose and s < 5)
            printif("        b: %.4g  => %.4g"%(tmp_b, srcs[s].b), 
                    verbose and s < 5)

            # sample location
            tmp_u = srcs[s].u
            u, ll = slicesample(xx       = srcs[s].u,
                                llh_func = loc_like,
                                step     = [1., 1.], 
                                step_out = False)
            srcs[s].u = u
            printif("        u: (%.4g, %4g) => (%.4g, %.4g)"%(tmp_u[0], tmp_u[1], u[0], u[1]), 
                    verbose and s < 5)
    return None


def printif(statement, condition):
    if condition:
        print statement

if __name__=="__main__":
    pass


