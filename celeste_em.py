import planck
import numpy as np
from scipy.optimize import fmin 
from celeste import celeste_likelihood_multi_image, gen_src_prob_layers
from util.plot_util import compare_to_model
import matplotlib.pyplot as plt

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
            printif("      img %d eps %2.2f => %2.2f"%(i, eps_tmp, img.epsilon), verbose)

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
                X_tildes = np.zeros(len(imgs))
                I_ts     = np.zeros(len(imgs))
                for n, img in enumerate(imgs):
                    # compute sum x_{n,m} p_{n,m,s}
                    X_tildes[n] = np.sum(all_src_probs[n][s+1,:,:] * img.nelec)

                    # compute I(t_s, beta_n)
                    I_ts[n] = planck.photons_per_joule(temp, img.band)
                return X_tildes, I_ts

            # profile likelihood - only of temperature
            def partial_loss(temp): 
                X_tildes, I_ts = compute_image_statistics(temp)
                return X_tildes.dot(np.log(I_ts)) - \
                       np.log(I_ts.sum()) * X_tildes.sum()

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
            X_tildes, I_ts = compute_image_statistics(t_hat)
            fac   = 1./(planck.lens_area * planck.exposure_duration * \
                        planck.sun_wattage / (planck.m_per_ly**2))
            b_hat = fac * (1./I_ts.sum()) * X_tildes.sum()

            ## 3) maximize u_s given \hat b_s, \hat t_s
            temp_tmp = srcs[s].t
            temp_b   = srcs[s].b
            srcs[s].t = t_hat
            srcs[s].b = b_hat
            printif("   src %d temp       = %2.2f => %2.2f"%(s, temp_tmp, srcs[s].t), verbose)
            printif("   src %d brightness = %2.2f => %2.2f"%(s, temp_b, srcs[s].b), verbose)

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

def printif(statement, condition):
    if condition:
        print statement

if __name__=="__main__":
    pass


