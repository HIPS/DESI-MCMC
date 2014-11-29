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

def testBirth():
    np.random.seed(42)

    ##
    ## Generate some fake sources using real image data (PSF and stuff)
    ##
    cat_glob = glob('../data/stamp_catalog/cat*.fits')[1:2]
    srcs, imgs, teff_catalog, us = load_imgs_and_catalog(cat_glob)
    print "initialized with %d sources and %d images"%(len(srcs), len(imgs))

    ## pixel location of source in image 0
    u_pixel = imgs[0].equa2pixel(srcs[0].u)

    ## generate fake sources information
    t_gt = np.array([5500])         # synthetic temperatures
    b_gt = np.array([5e-10])       # synthetic brightnesses
    gt_srcs = []
    for s in range(1):
        u_equa  = imgs[0].pixel2equa(u_pixel)
        src_s = celeste.PointSrcParams(u = u_equa, b = b_gt[s], t = t_gt[s])
        gt_srcs.append(src_s)

    print "Source at ", gt_srcs[0].u
    print "Brightness ", gt_srcs[0].b

    # re-generate images using these source params
    for img in imgs: 
        mimg      = celeste.gen_model_image(gt_srcs, img)
        img.nelec = np.random.poisson(mimg)

    # check out the first image
    #compare_to_model(gt_srcs, imgs[0])
    #plt.show()

    np.random.seed(42)
    rand = np.random.RandomState()
    Niter = 1
    e_samps  = np.zeros((Niter, len(imgs)))  # num images is fixed
    ll_samps = np.zeros(Niter)
    srcs = []
    post_samps = [srcs]                      # 
    for iter_n in range(Niter):
        print "Doing iteration", iter_n

        # fresh new universe model
        srcs = copy.deepcopy(srcs)
        srcs = birthStar(srcs, imgs[0], rand=rand)

        # keep track of universe model and image specific noise params
        post_samps.append(srcs)
        for n_img in range(len(imgs)):
            e_samps[iter_n, n_img] = imgs[n_img].epsilon
        ll_samps[iter_n] = celeste.celeste_likelihood_multi_image(srcs, imgs)
        print "After iteration %s: %s, cat len %s" % (iter_n, ll_samps[iter_n], len(srcs))
        print "Location: ", [src.u for src in srcs]
        print "Brightness: ", [src.b for src in srcs]


def testDeath():
    np.random.seed(42)

    ##
    ## Generate some fake sources using real image data (PSF and stuff)
    ##
    cat_glob = glob('../data/stamp_catalog/cat*.fits')[1:2]
    srcs, imgs, teff_catalog, us = load_imgs_and_catalog(cat_glob)
    print "initialized with %d sources and %d images"%(len(srcs), len(imgs))

    ## pixel location of source in image 0
    u_pixel = imgs[0].equa2pixel(srcs[0].u)

    ## generate fake sources information
    gt_srcs = []

    ## generate a fake source (that we want to die)
    t_new = np.array([5500])         # synthetic temperatures
    b_new = np.array([5e-10])       # synthetic brightnesses
    for s in range(1):
        us_equa  = imgs[0].pixel2equa(u_pixel)
        src_s = celeste.PointSrcParams(u = us_equa, b = b_new[s], t = t_new[s])
        #srcs.append(src_s)

    # re-generate images using these source params
    for img in imgs: 
        mimg      = celeste.gen_model_image(gt_srcs, img)
        img.nelec = np.random.poisson(mimg)

    # check out the first image
    #ll_trace, conv = celeste_em.celeste_em(srcs, imgs, maxiter=40, debug=False, verbose=1)
    #compare_to_model(gt_srcs, imgs[0])
    #plt.show()

    ##
    ## Initialize a single source with EM
    ##
    np.random.seed(42)
    rand = np.random.RandomState()
    Niter = 1
    e_samps  = np.zeros((Niter, len(imgs)))  # num images is fixed
    ll_samps = np.zeros(Niter)
    post_samps = [srcs]                      # 
    for iter_n in range(Niter):

        print "number of sources: ", len(srcs)
        # fresh new universe model
        srcs = deathStar(srcs, imgs[0], rand=rand)

        # keep track of universe model and image specific noise params
        post_samps.append(srcs)
        for n_img in range(len(imgs)):
            e_samps[iter_n, n_img] = imgs[n_img].epsilon
        ll_samps[iter_n] = celeste.celeste_likelihood_multi_image(srcs, imgs)

if __name__=="__main__":
    testBirth()
    testDeath()
