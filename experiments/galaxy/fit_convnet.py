import numpy as np
import os
from CelestePy import FitsImage
from glob import glob
import io_util as io
import matplotlib.pyplot as plt

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
    train_images, train_labels, test_images, test_labels = data
    train_images = add_color_channel(train_images) / 255.0
    test_images  = add_color_channel(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
    loss_grad = grad(loss_fun)

    # Initialize weights
    rs = npr.RandomState()
    W = rs.randn(N_weights) * param_scale

    # Check the gradients numerically, just to be safe
    # quick_grad_check(loss_fun, W, (train_images[:50], train_labels[:50]))

    print("    Epoch      |    Train err  |   Test error  ")
    def print_perf(epoch, W):
        test_perf  = frac_err(W, test_images, test_labels)
        train_perf = frac_err(W, train_images, train_labels)
        print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    # Train with sgd
    batch_idxs = make_batches(N_data, batch_size)
    cur_dir = np.zeros(N_weights)

    for epoch in range(num_epochs):
        print_perf(epoch, W)
        for idxs in batch_idxs:
            grad_W = loss_grad(W, train_images[idxs], train_labels[idxs])
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
            W -= learning_rate * cur_dir   


    # examine output
    



