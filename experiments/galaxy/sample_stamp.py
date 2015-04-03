import numpy as np
import sys, re, copy
from glob import glob
import os.path
from CelestePy import FitsImage, celeste_likelihood_multi_image, gen_model_image, PointSrcParams
from CelestePy.celeste_em import celeste_em
from CelestePy.celeste_mcmc import celeste_gibbs_sample
import CelestePy.planck as planck
from CelestePy.util.misc import init_utils, plot_util, check_grad
import CelestePy.celeste_galaxy_conditionals as gal

SAMPLE_FIELDS = ['epsilon', 'srcs', 'll']
def sample_source_params(srcs, imgs, Niter = 10, monitor=False):

    ## keep figure around for likelihood and param monitoring of source 0
    if monitor:
        fig, axarr = plt.subplots(2, 3)
        plt.ion()

    # source specific parameters
    src_samps = np.zeros((Niter, len(srcs)), dtype = PointSrcParams.src_dtype)
    e_samps   = np.zeros((Niter, len(imgs)))
    ll_samps  = np.zeros(Niter)

    # condition on source type for now
    for s,src in enumerate(srcs):
        src_samps[:,s]['a'].fill(src.a)

    prev_ll = celeste_likelihood_multi_image(srcs, imgs)
    print prev_ll
    for n in range(Niter):
        if n%1==0:
            print "===== iter %d of %d (curr_ll = %2.2f)==="%(n, Niter, prev_ll)

        if n%100==0:
            samp_dict = { 'epsilon' : e_samps,
                          'srcs'    : src_samps,
                          'll'      : ll_samps }
            save_samples(samp_dict, fname='tmp_samples.bin')
 
        # run a gibbs step
        if n > 0:
            celeste_gibbs_sample(srcs, imgs, subiter=1, verbose=False, debug=False)

        # save current likelihood and samples
        ll_samps[n] = celeste_likelihood_multi_image(srcs, imgs)
        prev_ll     = ll_samps[n]
        for n_img in range(len(imgs)):
            e_samps[n, n_img] = imgs[n_img].epsilon
        for s in range(len(srcs)):
            src_samps[n, s] = src_obj_to_array(srcs[s])
            print src_samps[n, s]

        # plot model image, true image comparison
        if monitor:
            print_samp(src_samps[n,0])
            model_image = gen_model_image(srcs[0:1], imgs[2])
            plot_util.compare_pair(imgs[2].nelec, model_image, axarr=axarr[0,0:3], standardize=True)
            plt.draw()

    samp_dict = { 'epsilon' : e_samps,
                  'srcs'    : src_samps,
                  'll'      : ll_samps }
    return samp_dict

def src_obj_to_array(src):
    """ returns a structured array """
    src_array = np.zeros(1, dtype = PointSrcParams.src_dtype)
    src_array['a'][0] = src.a
    if src.a == 0:
        src_array['t'][0] = src.t
        src_array['b'][0] = src.b
        src_array['u'][0] = src.u
    elif src.a == 1:
        src_array['u'][0]      = src.u
        src_array['v'][0]      = src.v
        src_array['theta'][0]  = src.theta
        src_array['phi'][0]    = src.phi
        src_array['sigma'][0]  = src.sigma
        src_array['rho'][0]    = src.rho
        src_array['fluxes'][0] = np.array([src.fluxes[b] for b in ['u', 'g', 'r', 'i', 'z']])
    return src_array

def array_to_src_obj(src_array):
    return PointSrcParams( 
        u = src_array['u'],
        a = src_array['a'],
        b = src_array['b'],
        t = src_array['t'],
        v = src_array['v'],
        theta  = src_array['theta'],
        phi    = src_array['phi'],
        sigma  = src_array['sigma'],
        rho    = src_array['rho'],
        fluxes = dict(zip(['u', 'g', 'r', 'i', 'z'], src_array['fluxes']))
        )

def print_samp(th):
    #theta, sigma, phi, rho = gal.constrain_params(th[0:4])
    print "    loc                : %2.2f, %2.2f"%(th['u'][0], th['u'][1])
    print "    r-flux             : %2.2f"%(th['fluxes'][2])
    print "    theta (prop dev)   : %2.5f"%(1.0 - th['theta'])
    print "    sigma (radius)     : %2.5f"%(th['sigma'])
    print "    phi   (angle, deg) : %2.5f"%(th['phi'] / np.pi * 180.)
    print "    rho   (major/minor): %2.5f"%(th['rho'])

# write samps to file
def save_samples(samp_dict, fname):
    f = file(fname, "wb")
    for s in SAMPLE_FIELDS:
        np.save(f, samp_dict[s])
    f.close()

def load_samples(fname):
    f = file(fname, "rb")
    samp_dict = {}
    for s in SAMPLE_FIELDS:
        samp_dict[s] = np.load(f)
    return samp_dict

if __name__=="__main__":

    ##
    ## Grab images and catalog data
    ##
    ## CAT 11 has one star, one gal (it looks like)
    cat_glob = glob('data/experiment_stamps/cat*.fits')[3:4]
    #cat_glob = glob('data/galaxy_stamps/cat*.fits')[3:4]
    cat_srcs, imgs, teff_catalog, us = init_utils.load_imgs_and_catalog(cat_glob)

    ## create srcs images
    srcs = init_utils.init_sources_from_image_block(imgs[0:5])[0:1]
    srcs[0] = init_utils.init_random_galaxy(srcs[0].u)
    srcs[0].phi   = .5 #0001
    srcs[0].sigma = 1. #0001
    srcs[0].rho   = .7 #.0001
    srcs[0].theta = .5 #.0001
    srcs[0].fluxes = cat_srcs[1].fluxes
    print "Initialized: "
    print "    %d images "%len(imgs)
    print "    %d catalog sources"%len(cat_srcs)
    print "    %d bright sources"%len(srcs)

    # visualize peaks!
    if False:
        plt.ion()
        plt.imshow(imgs[3].nelec.T, origin='lower')
        for src in srcs:
            peak = imgs[3].equa2pixel(src.u) - 1
            plt.scatter(peak[0], peak[1], s=20)

    #### initialize single source sample and image noises
    #fake_zs = []
    #fake_fluxes = {}
    #for img in imgs: 
    #    denoised = img.nelec - img.epsilon
    #    denoised[denoised <= 0] = 0
    #    fake_zs.append(denoised)
    #    fake_fluxes[img.band] = denoised.sum() / img.kappa * img.calib #10. #denoised.sum()
    #th = np.array([srcs[0].theta, srcs[0].sigma, srcs[0].phi, srcs[0].rho])
    #th = np.array([0., -1., 0., 0.])
    #th = np.concatenate((th, srcs[0].u))
    #th = np.concatenate((th, [fake_fluxes[b] for b in ['u', 'g', 'r', 'i', 'z']]))
    #print gal.galaxy_source_like(th, fake_zs, imgs)
    #print gal.galaxy_source_like_grad(th, fake_zs, imgs)
    #check_grad(lambda th: gal.galaxy_source_like(th, fake_zs, imgs),
    #           lambda th: gal.galaxy_source_like_grad(th, fake_zs, imgs),
    #           th)

    #plt.ion()
    #cur_dir = np.zeros(th.shape)
    #momentum = .98
    #learning_rate = 1e-5
    #lls = []
    #th_max = np.zeros(th.shape)
    #for i in range(20):
    #    th_grad = gal.galaxy_source_like_grad(th, fake_zs, imgs)
    #    cur_dir = momentum * cur_dir + (1.0 - momentum) * th_grad
    #    th      += learning_rate * cur_dir
    #    th[4:6] = srcs[0].u
    #    lls.append(gal.galaxy_source_like(th, fake_zs, imgs))
    #    print "ll = ", lls[-1]
    #    print th_grad

    #    # cache the ma
    #    if lls[-1] >= max(lls): 
    #        th_max = th.copy()
    #    print_prog(th, i)
    #    plt.plot(lls)
    #    plt.draw()

    #srcs[0].theta, srcs[0].sigma, srcs[0].phi, srcs[0].rho = \
    #    gal.constrain_params(th_max[0:4])
    #srcs[0].fluxes = dict(zip(['u', 'g', 'r', 'i', 'z'], th_max[-5:]))
    #srcs[0].u      = th_max[4:6]

    # run simple optmizer for galaxy params
    #import scipy.optimize as opt
    #res = opt.minimize(
    #    fun = lambda th: -1*galaxy_source_like(th, fake_zs, imgs),
    #    jac = lambda th: -1*galaxy_source_like_grad(th, fake_zs, imgs),
    #    x0  = th_max, 
    #    method = 'L-BFGS-B',
    #    options = {'disp': True}
    #    )
    #srcs[0].theta, srcs[0].sigma, srcs[0].phi, srcs[0].rho = transform_params(res.x[0:4])
    #srcs[0].fluxes = dict(zip(['u', 'g', 'r', 'i', 'z'], res.x[-5:]))
    #srcs[0].u      = th_max[4:6]


    # compare likelihoods
    #ml = celeste_likelihood_multi_image(srcs, imgs)
    #srcs[0].fluxes['r'] = 75.
    #ml_fix = celeste_likelihood_multi_image(srcs, imgs)
    #print "ML vs ML_fix: %2.3f vs. %2.3f"%(ml, ml_fix)
    #print "  mlfix better? ", ml_fix > ml

    if False:
        #cache initial likelihood, initial temp
        ll0 = celeste_likelihood_multi_image(srcs, imgs)
        model_image = gen_model_image(srcs[0:1], imgs[3])

        plot_util.compare_pair(imgs[3].nelec, model_image, standardize=False)
        plt.imshow(model_image, origin='lower'); plt.colorbar()
        plt.show()

    ##
    ## randomly initalize sky noise
    ##
    for img in imgs:
        img.epsilon = np.random.rand()*1e3

    ##
    ## Generate Point Source Param samples
    ##

    #print "========= EM for %d sources, %d images ================"%(len(srcs), len(imgs))
    #%prun -s tottime ll_trace, conv = celeste_em(srcs, imgs, 2, debug=False, verbose=1)
    #ll_trace, conv = celeste_em(srcs, imgs, maxiter=40, debug=False, verbose=1)
    #plot_util.compare_pair(imgs[2].nelec, gen_model_image(srcs[1:], imgs[2]))

    ##prun -s tottime samp_dict = sample_source_params(srcs, imgs, 2)
    samp_dict = sample_source_params(srcs, imgs, 
                                     Niter=100,
                                     monitor=True,
                                     th0 = )
    save_samples(samp_dict, "celeste_gal_samps.bin")

    ## visualize some model images visualization 
    #Nsamps = samp_dict['u'].shape[0]
    #src_samps = samp_dict['srcs']
    #for i in range(500, Nsamps):
    #    srcs  = [array_to_src_obj(sarray) for sarray in src_samps[i]]
    #    img_i = gen_model_image(srcs, imgs[2])
    #    plot_util.compare_pair(imgs[2].nelec, img_i); plt.colorbar()
    #    plt.imshow(img_i, origin='lower'); plt.colorbar()


