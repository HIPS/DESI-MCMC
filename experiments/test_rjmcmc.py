import numpy as np
import matplotlib.pyplot as plt
import sys, re, copy
from glob import glob
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('..'), os.path.pardir)))
import celeste
import celeste_em
import planck
from util.init_utils import load_imgs_and_catalog
from util.plot_util import compare_to_model
from mcmc_transitions import mergeStar, splitStar, birthStar, deathStar

if __name__=="__main__":
    np.random.seed(42)

    ##
    ## Generate some fake sources using real image data (PSF and stuff)
    ##
    cat_glob = glob('data/stamp_catalog/cat*.fits')[1:2]
    srcs, imgs, teff_catalog, us = load_imgs_and_catalog(cat_glob)
    print "initialized with %d sources and %d images"%(len(srcs), len(imgs))

    ## pixel location of source in image 0
    u_pixel = imgs[0].equa2pixel(srcs[0].u)

    ## generate fake sources information
    t_gt = np.array([5500, 8500])         # synthetic temperatures
    b_gt = np.array([5e-10, 8e-10])       # synthetic brightnesses
    gt_srcs = []
    for s in range(2):
        us_pixel = u_pixel + 3*np.random.randn(2)
        us_equa  = imgs[0].pixel2equa(us_pixel)
        src_s = celeste.PointSrcParams(u = us_equa, b = b_gt[s], t = t_gt[s])
        gt_srcs.append(src_s)

    # re-generate images using these source params
    for img in imgs: 
        mimg      = celeste.gen_model_image(gt_srcs, img)
        img.nelec = np.random.poisson(mimg)

    # check out the first image
    compare_to_model(gt_srcs, imgs[0])
    plt.show()

    ##
    ## Initialize a single source with EM
    ##
    np.random.seed(42)
    for s in srcs:
        s.t = np.random.rand()*9e3 + 1000
        s.b = np.random.rand()*1e-9
        s.fluxes = None
    for img in imgs:
        img.epsilon = np.random.rand()*1e3

    print "========= EM for %d sources, %d images ================"%(len(srcs), len(imgs))
    #%prun -s tottime ll_trace, conv = celeste_em(srcs, imgs, 2, debug=False, verbose=1)
    ll_trace, conv = celeste_em.celeste_em(srcs, imgs, maxiter=40, debug=False, verbose=1)
    compare_to_model(srcs, imgs[0])
    plt.show()

    ##
    ## Draw variable-dimension posterior samples
    ##
    rand = np.random.RandomState()
    Niter = 100
    e_samps  = np.zeros((Niter, len(imgs)))  # num images is fixed
    ll_samps = np.zeros(Niter)
    post_samps = [srcs]                      # 
    for iter_n in range(Niter):

        # fresh new universe model
        srcs = copy.deepcopy(srcs)

        # sample source params, conditioned on # srcs
        celeste_em.celeste_gibbs_sample(srcs, imgs, subiter=2, debug=False, verbose=True)

        im = imgs[0]
        if rand.rand() > 0.5:
            srcs = splitStar(srcs, im, rand=rand)
        else:
            srcs = mergeStar(srcs, im, rand=rand)

        if rand.rand() > 0.5:
            srcs = birthStar(srcs, im, rand=rand)
        else:
            srcs = deathStar(srcs, im, rand=rand)

        logprobs[it+1] = logprob = celeste_likelihood(srcs, im)
        print "After iteration %s: %s, cat len %s" % (it, logprob, len(srcs))
        print "Brightnesses:", [src.b for src in srcs]

        # keep track of universe model and image specific noise params
        post_samps.append(srcs)
        for n_img in range(len(imgs)):
            e_samps[iter_n, n_img] = imgs[n_img].epsilon
        ll_samps[n] = celeste_likelihood_multi_image(srcs, imgs)



