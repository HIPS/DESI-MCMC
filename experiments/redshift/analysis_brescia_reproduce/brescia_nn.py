from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer,FullConnection
import fitsio
import numpy as np
import time
import pickle

import sys
sys.path.insert(0, '../')
from redshift_utils import nanomaggies2mags, mags2nanomaggies

def brescia_nn(train, test, max_epochs=None, verbose=False):
    trainval_ds = SupervisedDataSet(5, 1)
    test_ds = SupervisedDataSet(5, 1)
    
    for datum in train:
        trainval_ds.addSample(datum[:5], (datum[5],))

    for datum in test:
        test_ds.addSample(datum[:5], (datum[5],))
    
    train_ds, val_ds = trainval_ds.splitWithProportion(0.75)
    
    if verbose:
        print "Train, validation, test:", len(train_ds), len(val_ds), len(test_ds)
    
    ns = {}
    min_error = -1
    min_h = -1
    
    # use validation to form 4-layer network with two hidden layers,
    # with (2n + 1) nodes in the first hidden layer and somewhere from
    # 1 to (n - 1) in the second hidden layer
    for h2 in range(1, 5):
        if verbose:
            start = time.time()
            print "h2 nodes:", h2
    
        # create the network
        if verbose:
            print "building network"

        n = FeedForwardNetwork()
        inLayer = LinearLayer(5)
        hiddenLayer1 = SigmoidLayer(11)
        hiddenLayer2 = SigmoidLayer(h2)
        outLayer = LinearLayer(1)
    
        n.addInputModule(inLayer)
        n.addModule(hiddenLayer1)
        n.addModule(hiddenLayer2)
        n.addOutputModule(outLayer)
    
        in_to_hidden = FullConnection(inLayer, hiddenLayer1)
        hidden_to_hidden = FullConnection(hiddenLayer1, hiddenLayer2)
        hidden_to_out = FullConnection(hiddenLayer2, outLayer)
    
        n.addConnection(in_to_hidden)
        n.addConnection(hidden_to_hidden)
        n.addConnection(hidden_to_out)
    
        n.sortModules()
    
        # training
        if verbose:
            print "beginning training"
        trainer = BackpropTrainer(n, train_ds)
        trainer.trainUntilConvergence(maxEpochs=max_epochs)

        ns[h2] = n
    
        # validation
        if verbose:
            print "beginning validation"

        out = n.activateOnDataset(val_ds)
        actual = val_ds['target']
        error = np.sqrt(np.sum((out - actual)**2) / len(val_ds))
        if verbose:
            print "RMSE:", error
    
        if min_error == -1 or error < min_error:
            min_error = error
            min_h = h2
    
        if verbose:
            stop = time.time()
            print "Time:", stop - start
    
    # iterate through
    if verbose:
        print "best number of h2 nodes:", min_h
    out_test = ns[min_h].activateOnDataset(test_ds)

    return ns[h2], out_test

if __name__ == '__main__':
    data_file = fitsio.FITS('../dr7qso.fit')[1].read()
    
    data = np.zeros((len(data_file['UMAG']), 6))
    data[:,0] = data_file['UMAG']
    data[:,1] = data_file['GMAG']
    data[:,2] = data_file['RMAG']
    data[:,3] = data_file['IMAG']
    data[:,4] = data_file['ZMAG']
    data[:,5] = data_file['z']

    # make sure there are no zero mags
    for i in range(5):
        data = data[data[:,i] != 0]

    # convert to nanomaggies for the sake of example
    data[:,:5] = mags2nanomaggies(data[:,:5])

    # split into training and test
    train = data[:int(0.8 * len(data)),:]
    test = data[int(0.8 * len(data)):,:]

    model, preds = brescia_nn(train, test, verbose=True)

    # calculate RMSE
    actual_test = test[:,5]
    rmse = np.sqrt(np.sum((preds - actual_test)**2) / len(test))

    output = open('brescia_output.pkl', 'wb')
    pickle.dump(model, output)
    output.close()

    print "RMSE:", rmse

