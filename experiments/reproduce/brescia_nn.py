from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer,FullConnection
import fitsio
import numpy as np

# constants to change
MAX_EPOCHS = 100
NUM_DATA = 100

# generate dataset
data_file = fitsio.FITS('dr7qso.fit')[1].read()

alldata = SupervisedDataSet(5, 1)
length = len(data_file['UMAG'])

for i in range(NUM_DATA):
    umag = data_file['UMAG'][i]
    gmag = data_file['GMAG'][i]
    rmag = data_file['RMAG'][i]
    imag = data_file['IMAG'][i]
    zmag = data_file['ZMAG'][i]
    redshift = data_file['z'][i]
    alldata.addSample((umag, gmag, rmag, imag, zmag), (redshift,))

trainval_ds, test_ds = alldata.splitWithProportion(0.8)
train_ds, val_ds = trainval_ds.splitWithProportion(0.75)

print "Train, validation, test:", len(train_ds), len(val_ds), len(test_ds)

ns = []
min_error = -1
min_h = -1

# use validation to form 4-layer network with two hidden layers,
# with (2n + 1) nodes in the first
for h2 in range(1, 5):
    print "h2 nodes:", h2

    # create the network
    print "building network"
    ns.append(FeedForwardNetwork())
    inLayer = LinearLayer(5)
    hiddenLayer1 = SigmoidLayer(11)
    hiddenLayer2 = SigmoidLayer(h2)
    outLayer = LinearLayer(1)

    n = ns[h2 - 1]
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
    print "beginning training"
    trainer = BackpropTrainer(n, train_ds)
    trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)

    # validation
    print "beginning validation"
    out = n.activateOnDataset(val_ds)
    actual = val_ds['target']
    error = np.sqrt(np.sum((out - actual)**2) / len(val_ds))
    print "RMSE:", error

    if min_error == -1 or error < min_error:
        min_error = error
        min_h = h2

# iterate through
print "best number of h2 nodes:", min_h
out_test = ns[min_h - 1].activateOnDataset(test_ds)
actual_test = test_ds['target']
print "Test RMSE", np.sqrt(np.sum((out_test - actual_test)**2) / len(test_ds))

