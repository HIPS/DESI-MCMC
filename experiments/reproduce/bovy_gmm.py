import fitsio
import numpy as np
from sklearn import mixture
import time
import pickle

CACHED = False

# import FITS file
data_file = fitsio.FITS('dr7qso.fit')[1].read()

# change to relative fluxes
data = np.zeros((len(data_file['UMAG']), 6))
data[:,0] = data_file['IMAG']
data[:,1] = data_file['UMAG']
data[:,2] = data_file['GMAG']
data[:,3] = data_file['RMAG']
data[:,4] = data_file['ZMAG']
data[:,5] = data_file['z']

data = data[data[:,0] != 0]
for i in range(1, 5):
    data[:,i] /= data[:,0]

# split into training, validation, and test
train = data[:int(0.6 * len(data)),:]
val = data[int(0.6 * len(data)):int(0.8 * len(data)),:]
test = data[int(0.8 * len(data)):,:]


max_n = -1
max_score = np.inf

# loop over number of Gaussians in mixture
# find which is best through validation likelihood
diff = 0.2
min_i = 17.5
max_i = 20.5

len_test = len(test)
test_below = len(test[test[:,0] < min_i])
test_above = len(test[test[:,0] > max_i])

if not CACHED:
    for n in range(60):
        print "testing Gaussian with components:", n
        score = 0.
        for bin in np.arange(min_i, max_i, diff):
            print "processing bin", bin
            train_bin = train[train[:,0] > bin]
            train_bin = train_bin[train_bin[:,0] < (bin + diff)]
    
            g = mixture.GMM(n_components=n, covariance_type='full')
            g.fit(train_bin[:,1:])
    
            val_bin = val[val[:,0] > bin]
            val_bin = val_bin[val_bin[:,0] < (bin + diff)]
            score += np.sum(g.score(val_bin[:,1:]))
    
        print "score", score
        if score > max_score:
            max_score = score
            max_n = n
    
    print "best n is:", max_n

max_n = 25

if CACHED:
    pkl_file = open('bovy.pkl')
    gs = pickle.load(pkl_file)
else:
    gs = []

# test on test set (calculate mean and MLE)
for i,bin in enumerate(np.arange(min_i, max_i, diff)):
    start = time.time()

    print "on bin", bin
    train_bin = train[train[:,0] > bin]
    train_bin = train_bin[train_bin[:,0] < (bin + diff)]

    if CACHED:
        print "reading model from cache"
        g = gs[i]
    else:
        g = mixture.GMM(n_components=max_n, covariance_type='full')
        g.fit(train_bin[:,1:])

        gs.append(g)

    test_bin = test[test[:,0] > bin]
    test_bin = test_bin[test_bin[:,0] < (bin + diff)]

    # calculate mean and mle
    sum_sq_mle = 0.
    sum_sq_mean = 0.

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
        sum_sq_mle += (z_max - t[4])**2
        sum_sq_mean += (mean - t[4])**2

    stop = time.time()
    print "time for bin:", stop - start

print "mle RMSE", np.sqrt(sum_sq_mle / (len_test - test_above - test_below))
print "mean RMSE", np.sqrt(sum_sq_mean / (len_test - test_above - test_below))

if not CACHED:
    output = open('bovy.pkl', 'wb')
    pickle.dump(gs, output)
    output.close()

