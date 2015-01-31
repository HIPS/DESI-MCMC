import sys, re, copy
from glob import glob
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('..'), os.path.pardir)))
from celeste import FitsImage, celeste_likelihood_multi_image, gen_model_image
from celeste_em import celeste_em, celeste_gibbs_sample
import planck
from util.init_utils import load_imgs_and_catalog
import numpy as np
import matplotlib.pyplot as plt

def sample_source_params(srcs, imgs, Niter = 10, fname='star_samples_rand.bin'):
    e_samps = np.zeros((Niter, len(imgs)))    # sky term
    t_samps = np.zeros((Niter, len(srcs)))    # source temp
    b_samps = np.zeros((Niter, len(srcs)))    # source brightness
    u_samps = np.zeros((Niter, len(srcs), 2)) # source location 
    ll_samps = np.zeros(Niter)
    prev_ll = celeste_likelihood_multi_image(srcs, imgs)
    print prev_ll
    for n in range(Niter):
        if n%1==0:
            print "===== iter %d of %d (curr_ll = %2.2f)==="%(n, Niter, prev_ll)

        if n%100==0:
            save_samples(e_samps, t_samps, b_samps, u_samps, ll_samps, fname=fname)

        # run a gibbs step
        if n > 0:
            celeste_gibbs_sample(srcs, imgs, subiter=1, verbose=False, debug=False)

        # save current likelihood and samples
        ll_samps[n] = celeste_likelihood_multi_image(srcs, imgs)
        prev_ll = ll_samps[n]
        for n_img in range(len(imgs)):
            e_samps[n, n_img] = imgs[n_img].epsilon
        for s in range(len(srcs)):
            t_samps[n, s] = srcs[s].t
            b_samps[n, s] = srcs[s].b
            u_samps[n, s, :] = srcs[s].u

    # save and return
    save_samples(e_samps, t_samps, b_samps, u_samps, ll_samps, fname=fname)
    return t_samps, b_samps, u_samps, e_samps, ll_samps

def geweke_source_params(srcs, imgs, Niter = 10):
    e_samps = np.zeros((Niter, len(imgs)))    # sky term
    t_samps = np.zeros((Niter, len(srcs)))    # source temp
    b_samps = np.zeros((Niter, len(srcs)))    # source brightness
    u_samps = np.zeros((Niter, len(srcs), 2)) # source location 
    ll_samps = np.zeros(Niter)
    prev_ll = celeste_likelihood_multi_image(srcs, imgs)
    print prev_ll
    for n in range(Niter):
        if n%1==0:
            print "===== iter %d of %d (curr_ll = %2.2f)==="%(n, Niter, prev_ll)

        if n%100==0:
            save_samples(e_samps, t_samps, b_samps, u_samps, ll_samps, fname='star_samples_teff_seed.bin')

        # run a gibbs step
        if n > 0:
            celeste_gibbs_sample(srcs, imgs, subiter=1, verbose=False, debug=False)

        # save current likelihood and samples
        ll_samps[n] = celeste_likelihood_multi_image(srcs, imgs)
        prev_ll = ll_samps[n]
        for n_img in range(len(imgs)):
            e_samps[n, n_img] = imgs[n_img].epsilon
        for s in range(len(srcs)):
            t_samps[n, s] = srcs[s].t
            b_samps[n, s] = srcs[s].b
            u_samps[n, s, :] = srcs[s].u
    return t_samps, b_samps, u_samps, e_samps, ll_samps

# write samps to file
def save_samples(e, t, b, u, ll, fname="star_samples.bin"):
    f = file(fname, "wb")
    np.save(f,e)
    np.save(f,t)
    np.save(f,b)
    np.save(f,u)
    np.save(f,ll)
    f.close()

def load_samples(fname='star_samples.bin'):
    f = file(fname, 'rb')
    e = np.load(f)
    t = np.load(f)
    b = np.load(f)
    u = np.load(f)
    ll = np.load(f)
    f.close()
    return e, t, b, u, ll

if __name__=="__main__":

    ##
    ## Generate some fake sources using real image data (PSF and stuff)
    ##
    cat_glob = glob('data/stamp_catalog/cat*.fits')[0:20]
    srcs, imgs, teff_catalog, us = load_imgs_and_catalog(cat_glob)
    print "initialized with %d sources and %d images"%(len(srcs), len(imgs))

    ##
    ##  Initialize sources with Expectation Maximization
    ##
    # randomize source values (except, of course, locations)
    do_synthetic = False
    b_gt = np.zeros(len(srcs))
    if do_synthetic:
        np.random.seed(42)
        for s in srcs:
            s.t = np.random.rand()*9e3 + 1000
            s.b = np.random.rand()*1e-9
            s.fluxes = None
        for img in imgs:
            img.epsilon = np.random.rand()*1e3

        # save ground truth values
        t_gt = np.array([s.t for s in srcs])
        teff_catalog = t_gt
        b_gt = np.array([s.b for s in srcs])
        e_gt = np.array([img.epsilon for img in imgs])

        # re-generate images using these source params
        for img in imgs: 
            mimg      = gen_model_image(srcs, img)
            img.nelec = np.random.poisson(mimg)

    #cache initial likelihood, initial temp
    %lprun -m celeste  ll0 = celeste_likelihood_multi_image(srcs, imgs)
    print "Data generating log like: ", ll0

    # re-initialize source values randomly
    for s in srcs:
        s.t = np.random.rand()*9e3 + 1000
        s.b = np.random.rand()*1e-9
        s.fluxes = None

    for img in imgs:
        img.epsilon = np.random.rand()*1e3

    print "========= EM for %d sources, %d images ================"%(len(srcs), len(imgs))
    #%prun -s tottime ll_trace, conv = celeste_em(srcs, imgs, 2, debug=False, verbose=1)
    ll_trace, conv = celeste_em(srcs, imgs, maxiter=10, debug=False, verbose=1)
    for s in srcs:
        if s.t > 20000 or s.t < 200: 
            s.t = 10000
    srcs_init = copy.deepcopy(srcs)
    imgs_init = copy.deepcopy(imgs)

    # copy over EM init sources
    srcs = copy.deepcopy(srcs_init)
    imgs = copy.deepcopy(imgs_init)

    #############################################################
    ## SAMPLER CALL / or load from disk
    #############################################################
    fname='star_samples_rand.bin'

    %lprun -m util.slicesample -m celeste_em \
    t_samps, b_samps, u_samps, e_samps, ll_samps = \
        sample_source_params(srcs, imgs, Niter=2, fname=fname)

    if False:
        t_samps, b_samps, u_samps, e_samps, ll_samps = load_samples(fname)
    #############################################################

    # assert posterior means approach ground truth values
    sources_to_plot = [3, 4, 8]
    for s in sources_to_plot:
        fig, axarr = plt.subplots(1, 3)
        use_idx = range(250, t_samps.shape[0])

        # plot Temperature histograms
        hout = axarr[0].hist(t_samps[use_idx, s], 20, normed=True, alpha=.5)
        if teff_catalog[s] > 0:
            axarr[0].vlines(teff_catalog[s], hout[0].min(), hout[0].max(), color='green', linewidth=3)

        # plot brightness histograms
        hout = axarr[1].hist(b_samps[use_idx, s], 20, normed=True, alpha=.5)
        if b_gt[s] > 0:
            axarr[1].vlines(b_gt[s], hout[0].min(), hout[0].max(), color='green', linewidth=3)

        # plot joint t and brightness scatterplot
        axarr[2].scatter(t_samps[use_idx, s], b_samps[use_idx, s])
        axarr[2].set_ylim((b_samps[use_idx, s].min(), b_samps[use_idx, s].max()))
        #axarr[s].scatter(t_gt[s], b_gt[s], color='red', s=20)
        #axarr[s].vlines(b_gt[s], hout[0].min(), hout[0].max(), color='green', linewidth=3)
    plt.show()


