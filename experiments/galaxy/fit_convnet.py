import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.signal
from autograd import grad
from autograd.util import quick_grad_check
import os
from CelestePy import FitsImage
from glob import glob
import io_util as io
import conv_net as cv
import matplotlib.pyplot as plt
import gmm_util as gu
import seaborn as sns

def stamps2array(gstamps):
    stamps = []
    for gstamp in gstamps:
        template = gstamp.replace('-g-', '-%s-')
        bands = ['u', 'g', 'r', 'i', 'z']
        imgs = [FitsImage(b, fits_file_template=template) for b in bands]
        img_array = np.array([img.nelec for img in imgs])
        stamps.append( np.rollaxis(img_array, 0, 3) )
    stamps = np.array(stamps)
    return stamps

logit = lambda x: np.log(x) - np.log(1.-x)
def rec2matrix(samps):
    return np.column_stack((
        logit(samps['theta']),
        logit(samps['phi']/np.pi),
        np.log(samps['sigma']),
        samps['fluxes']))

def extract_stamp_ids(stamp_files):
    ids = ["-".join(os.path.splitext(g)[0].split('-')[-2:]) for g in stamp_files]
    return ids

def load_stamps_and_samps(gstamps):

    # gather all stamp files!
    print "loading available stamps"
    gstamps.sort()
    stamp_ids = extract_stamp_ids(gstamps)
    stamps    = stamps2array(gstamps)

    # gather all samps!
    print "loading MCMC sample files"
    gal_chain_template = 'samp_cache/run5/gal_samps_stamp_%s_chain_0.bin'
    gal_chain_files = [gal_chain_template%sid for sid in stamp_ids]
    chain_mask      = np.zeros(len(stamp_ids), dtype=np.bool)  # keep track of the ones that actually have samples
    Nselect = 500
    Nskip   = 5
    samps  = []
    for i,chain in enumerate(gal_chain_files):
        print "Galaxy", os.path.basename(chain)

        ## 0. load four chains from disk
        src_samp_chains, ll_samp_chains, eps_samp_chains = \
            io.load_mcmc_chains(chain, num_chains=4)

        if len(src_samp_chains) > 0:
            th            = rec2matrix( np.concatenate(src_samp_chains))
            # make sure there are no infinite samples
            if np.any(np.isinf(th)) or np.any(np.isnan(th)):
                continue
            chain_mask[i] = True
            samps.append(th[-Nselect*Nskip:-1:Nskip, :])

    print "There are %d chains with either missing, zeros, or otherwise unsuitable samples"%((~chain_mask).sum())

    # samps and stamps now aligned
    stamps = stamps[chain_mask, :, :, :]
    samps  = np.array(samps)
    return stamps, samps


if __name__=="__main__":

    # load in stamp/samp pairs
    gstamps = glob("stamps/*stamp-g*.fits")
    stamps, samps = load_stamps_and_samps(gstamps)
    print "Loaded %d stamp/samp pairs"%stamps.shape[0]

    # fit the damn thing
    train_images = np.rollaxis(stamps, 3, 1)
    N_data = train_images.shape[0]

    # Network parameters
    L2_reg = 100.
    input_shape = train_images.shape[1:]
    layer_specs = [#fast_conv_layer((5, 5), 6, input_shape[1:]),
                   cv.conv_layer((12, 12), 10),
                   cv.maxpool_layer((2, 2)),
                   #cv.conv_layer((5, 5), 16),
                   #fast_conv_layer((5, 5), 16, (12, 12)),
                   #cv.maxpool_layer((2, 2)),
                   cv.tanh_layer(300),
                   cv.linear_layer(100),
                   cv.linear_layer(16)]

    # Training parameters
    param_scale = 0.01
    learning_rate = 1e-7
    momentum = 0.8
    batch_size = 20
    num_epochs = 100

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = \
        cv.make_nn_funs(input_shape, layer_specs, L2_reg)
    loss_grad = grad(loss_fun)

    # Initialize weights
    rs = npr.RandomState()
    W = rs.randn(N_weights) * param_scale

    print loss_fun(W, train_images[:10], samps[:10])
    print 
    %lprun -m conv_net \
    loss_grad(W, train_images[:10], samps[:10])

    # Check the gradients numerically, just to be safe
    quick_grad_check(loss_fun, W, (train_images[:10], samps[:10]))

    ########################################################################
    # Plotting/Printing Funcs
    ########################################################################
    print("    Epoch      |    Train err  |   Best error ")
    best_w = W.copy()
    best_loss = np.inf
    def callback(x, i, g):
        train_perf = loss_fun(x, train_images, samps)
        global best_loss
        global best_w
        if train_perf < best_loss:
            best_w = x.copy()
            best_loss = train_perf
        print("{0:15}|{1:15}|{2:15}".format(i, "%2.5g"%train_perf, best_loss))

        if i % 5 == 0:
            plot_train_fits(best_w, axarr)

    def plot_train_fits(W, axarr):
        params    = pred_fun(W, train_images)
        means     = params[:, :8]
        variances = np.exp(params[:,-8:]) # axis aligned variances

        # plot 5 random data points and 4 random marginals
        idx  = np.sort(np.random.permutation(train_images.shape[0])[:axarr.shape[0]])
        dims = np.sort(np.random.permutation(samps.shape[-1])[:axarr.shape[1]])
        for r, i in enumerate(idx):
            for c, d in enumerate(dims):
                axarr[r, c].cla()
                n, bins, patches = axarr[r, c].hist(samps[i, :, d], 
                                                    bins=20, normed=True)
                axarr[r, c].plot( [means[i, d], means[i, d]], [0, n.max()] )
                thgrid = np.linspace(min(bins[0],  means[i,d]),
                                     max(bins[-1], means[i,d]), 50)
                axarr[r, c].plot(thgrid, 
                    np.exp(gu.mog_logmarglike(thgrid, 
                                   means = np.array([ means[i] ]),
                                   covs  = np.array([ np.diag(variances[i]) ]),
                                   pis   = np.array([1.]),
                                   ind   = d)))
                axarr[r, c].set_title("Idx = %d, dim = %d"%(i, d))
        plt.draw()

    ## interactive plot for tracking during learning
    plt.ion()
    fig, axarr = plt.subplots(5, 4, figsize=(13, 9))

    ## Create Batches, Call Optimizer
    batch_idxs = cv.make_batches(N_data, batch_size)
    cur_dir    = np.zeros(N_weights)
    means      = pred_fun(W, train_images)
    from optimizers import adam, rmsprop
    def sub_loss_grad(x, i):
        return loss_grad(x, 
                         train_images[batch_idxs[i%len(batch_idxs)]],
                         samps[batch_idxs[i%len(batch_idxs)]])
    W = rmsprop(sub_loss_grad,
                x = W, callback=callback,
                step_size = .006,
                gamma     = .6, 
                eps       = 1e-8, 
                num_iters = 200)

    ####################################################################
    #examine output
    ####################################################################
    print "Final Loss: ", loss_fun(best_w, train_images, samps)
    params    = pred_fun(best_w, train_images)
    means     = params[:, :8]
    variances = params[:, -8:]

    i = 10
    def compare_moments(i):
        print "samp comparison, idx = %d "%i
        print " {0:5} | {1:6} | {2:6} | {3:6} | {4:6} ".format(
                "dim", "mod_m", "sam_m", "mod_v", "sam_v")
        smean = samps[i].mean(axis=0)
        svar  = samps[i].var(axis=0)
        for i, (mm, mv, m, v) in enumerate(zip(means[i, :], variances[i, :], smean, svar)):
            print " {0:5} | {1:6} | {2:6} | {3:6} | {4:6} ".format(
                    i, "%2.2f"%mm, "%2.2f"%m, "%2.2f"%mv, "%2.2f"%v)

    compare_moments(0)
    compare_moments(10)
    compare_moments(80)


    ######### exploratory stuff - look at the scaling of each distribution
    svals = []
    for i in range(len(samps)):
        u, s, v = np.linalg.svd(samps[i])
        svals.append(s)
    svals = np.vstack(svals)

