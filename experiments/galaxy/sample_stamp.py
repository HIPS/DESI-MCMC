import numpy as np
import sys, re, copy
from glob import glob
import os.path
from CelestePy import FitsImage, SrcParams, \
                      celeste_likelihood_multi_image, gen_model_image
from CelestePy.celeste_em import celeste_em
from CelestePy.celeste_mcmc import celeste_gibbs_sample
import CelestePy.planck as planck
from CelestePy.util.misc import init_utils, plot_util, check_grad
import CelestePy.celeste_galaxy_conditionals as gal

SAMPLE_FIELDS = ['epsilon', 'srcs', 'll']
def sample_source_params(srcs, imgs, Niter = 10, monitor=False, plot=False, saveas="tmp_samples.bin"):

    ## keep figure around for likelihood and param monitoring of source 0
    if plot:
        fig, axarr = plt.subplots(2, 3)
        plt.ion()

    # source specific parameters
    src_samps = np.zeros((Niter, len(srcs)), dtype = SrcParams.src_dtype)
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
            save_samples(samp_dict, fname=saveas)

        # run a gibbs step
        if n > 0:
            celeste_gibbs_sample(srcs, imgs, subiter=1, verbose=False, debug=False)

        # save current likelihood and samples
        ll_samps[n] = celeste_likelihood_multi_image(srcs, imgs)
        prev_ll     = ll_samps[n]
        for n_img in range(len(imgs)):
            e_samps[n, n_img] = imgs[n_img].epsilon
        for s in range(len(srcs)):
            src_samps[n, s] = srcs[s].to_array()

        # plot model image, true image comparison
        if monitor:
            print_samp(src_samps[n,0])
        if plot:
            model_image = gen_model_image(srcs[0:1], imgs[2])
            plot_util.compare_pair(imgs[2].nelec, model_image, axarr=axarr[0,0:3], standardize=True)
            plt.draw()

    samp_dict = { 'epsilon' : e_samps,
                  'srcs'    : src_samps,
                  'll'      : ll_samps }
    return samp_dict

def print_samp(th):
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

    ##########################################################################
    ## set sampling parameters
    ##########################################################################
    narg    = len(sys.argv)
    stamp_n = int(sys.argv[1]) if narg > 1 else 0
    Nsamps  = int(sys.argv[2]) if narg > 2 else 20
    Nchains = int(sys.argv[3]) if narg > 3 else 2

    ##########################################################################
    ## Grab images, catalog data and initialize galaxy source
    ##########################################################################
    cat_glob = glob('data/experiment_stamps/cat*.fits')
    cat_glob.sort()
    cat_glob = cat_glob[stamp_n:(stamp_n+1)]
    cat_srcs, imgs, teff_catalog, us = init_utils.load_imgs_and_catalog(cat_glob)

    ## create srcs images
    srcs = init_utils.init_sources_from_image_block(imgs[0:5])[0:1]
    srcs[0]        = init_utils.init_random_galaxy(srcs[0].u)
    srcs[0].phi    = np.random.rand() * np.pi
    srcs[0].sigma  = np.random.rand()
    srcs[0].rho    = np.random.rand()
    srcs[0].theta  = np.random.rand()
    srcs[0].fluxes = cat_srcs[1].fluxes
    print "Initialized: "
    print "    %d images "%len(imgs)
    print "    %d catalog sources"%len(cat_srcs)
    print "    %d bright sources"%len(srcs)

    ## visualize peaks!
    if False:
        plt.ion()
        plt.imshow(imgs[3].nelec.T, origin='lower')
        for src in srcs:
            peak = imgs[3].equa2pixel(src.u) - 1
            plt.scatter(peak[0], peak[1], s=20)

    ##
    ## randomly initalize sky noise
    ##
    for img in imgs:
        img.epsilon = np.random.rand()*1e3

    ##
    ## Generate Point Source Param samples
    ##
    ##%lprun -m CelestePy.celeste_galaxy_conditionals sample_source_params(srcs, imgs, Niter=5, monitor=True)
    for chain_n in range(Nchains):
        stamp_id = os.path.splitext(os.path.basename(cat_glob[0]))[0][4:]
        out_name = "gal_samps_stamp_%s_chain_%d.bin"%(stamp_id, chain_n)
        print "==========================================================="
        print "====== RUNNING CHAIN %d ==================================="%chain_n
        print "=== saving samples as ", out_name
        print "==========================================================="
        samp_dict = sample_source_params(srcs, imgs,
                                         Niter   = Nsamps,
                                         monitor = True,
                                         saveas  = out_name)
        save_samples(samp_dict, out_name)


