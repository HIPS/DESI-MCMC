import fitsio
import numpy as np
from sklearn import mixture
import time
import pickle

import sys
sys.path.insert(0, '../')
from redshift_utils import nanomaggies2mags, mags2nanomaggies

"""
train and test have 6 columns, 5 ugriz bands in nanomaggies + redshift
min_i, max_i, diff - in units of mags for binning i-magnitude
max_gaussians - the maximum number of components in the mixture you want
to validate
verbose - should we print debugging output?
"""
def bovy_xdqsoz(train_raw, test_raw, min_i, max_i, diff, max_gaussians,
                verbose=False):
    train = np.copy(train_raw)
    test = np.copy(test_raw)

    # rearrange to get IUGRZz
    for data,raw in [(train,train_raw), (test,test_raw)]:
        data[:,0] = raw[:,3]
        data[:,1] = raw[:,0]
        data[:,2] = raw[:,1]
        data[:,3] = raw[:,2]

    for i in range(1, 5):
        train[:,i] /= train[:,0] 
        test[:,i] /= test[:,0] 

    train[:,0] = nanomaggies2mags(train[:,0])
    test[:,0] = nanomaggies2mags(test[:,0])

    val = train[(0.75 * len(train)):,:]
    train = train[:(0.75 * len(train)),:]

    if verbose:
        print "Sizes:", len(train), len(val), len(test)

    max_n = -1
    max_score = -np.inf

    # loop over number of Gaussians in mixture
    # find which is best through validation likelihood
    for n in range(1, max_gaussians):
        if verbose:
            print "testing Gaussian with components:", n

        score = 0.
        for bin in np.arange(min_i, max_i, diff):
            if verbose:
                print "processing bin", bin
            train_bin = train[train[:,0] > bin]
            train_bin = train_bin[train_bin[:,0] < (bin + diff)]
            g = mixture.GMM(n_components=n, covariance_type='full')
            g.fit(train_bin[:,1:])

            val_bin = val[val[:,0] > bin]
            val_bin = val_bin[val_bin[:,0] < (bin + diff)]
            score += np.sum(g.score(val_bin[:,1:]))
        if verbose:
            print "score", score

        if score > max_score:
            max_score = score
            max_n = n

    if verbose:
        print "best n is:", max_n

    gs = []

    if verbose:
        print "beginning testing"

    # test on test set (calculate mean and MLE)
    test = np.hstack([test, np.zeros((len(test), 2))])
    for bin in np.arange(min_i, max_i, diff):
        if verbose:
            start = time.time()
            print "on bin", bin

        train_bin_idx = np.where((train[:,0] > bin) & (train[:,0] <= (bin+diff)))[0]
        train_bin = train[train_bin_idx, :]
        #train_bin = train[train[:,0] > bin]
        #train_bin = train_bin[train_bin[:,0] < (bin + diff)]

        g = mixture.GMM(n_components=max_n, covariance_type='full')
        g.fit(train_bin[:,1:])

        gs.append(g)

        test_idx = (test[:,0] > bin) & (test[:,0] < (bin + diff))
        test_bin = test[test_idx]

        # calculate mean and mle
        sum_sq_mle = 0.
        sum_sq_mean = 0.
        preds = np.zeros((len(test_bin), 2))
        for i,t in enumerate(test_bin):
            max_score_test = -np.inf
            z_max = -1
            mean = 0.
            sum_weights = 0.
            for z in np.arange(0, 6, 0.01):
                test_copy = np.zeros((1, 5))
                test_copy[0,:4] = t[1:5]
                test_copy[0, 4] = z

                score = g.score(test_copy)
                if score > max_score_test:
                    max_score_test = score
                    z_max = z

                prob = np.exp(score)
                sum_weights += prob
                mean += prob * z

            mean /= sum_weights
            preds[i][0] = z_max
            preds[i][1] = mean

        test[test_idx,6:8] = preds[:,:]
    
        if verbose:
            stop = time.time()
            print "time for bin:", stop - start
    
    return gs, test[:,6], test[:,7]

if __name__ == "__main__":
    # import FITS file
    data_file = fitsio.FITS('../../../data/DR10QSO/DR10Q_v2.fits')[1].read()
    
    #data = np.zeros((len(data_file['UMAG']), 6))
    data = np.column_stack((data_file['PSFMAG'], data_file['Z_VI']))
    data = data[0:1000,:]
    #data[:,0] = data_file['UMAG']
    #data[:,1] = data_file['GMAG']
    #data[:,2] = data_file['RMAG']
    #data[:,3] = data_file['IMAG']
    #data[:,4] = data_file['ZMAG']
    #data[:,5] = data_file['z']
    

    
    # convert to nanomaggies for the sake of example
    data[:,:5] = mags2nanomaggies(data[:,:5])

    # make sure there are no zero mags
    for i in range(5):
        data = data[data[:,i] != 0]

    valid_idx = np.sum(np.isinf(data), axis=1) == 0
    data      = data[valid_idx, :]



    # split into training and test
    train = data[:int(0.8 * len(data)),:]
    test = data[int(0.8 * len(data)):,:]

    # minimum and maximum MAGS we want to bin
    min_i = 17.5
    max_i = 20.5
    width_i = 1.
    max_gaussians = 10

    model, preds_mle, preds_mean = \
        bovy_xdqsoz(train, test, min_i, max_i, width_i, max_gaussians, verbose=True)

    # don't include indices for which we did not make a prediction, i.e.,
    # i-magnitude out of range
    pred_idx = preds_mle != 0

    preds_mle = preds_mle[pred_idx]
    preds_mean = preds_mean[pred_idx]
    actual_test = test[pred_idx,5]

    rmse_mle = np.sqrt(np.sum((preds_mle - actual_test)**2) / len(actual_test))
    rmse_mean = np.sqrt(np.sum((preds_mean - actual_test)**2) / len(actual_test))

    output = open('bovy_output.pkl', 'wb')
    pickle.dump(model, output)
    output.close()

    print "mean RMSE:", rmse_mean
    print "mle RMSE:", rmse_mle

