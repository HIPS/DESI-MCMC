import os, matplotlib
r = os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"')
if r != 0:
  matplotlib.use('Agg')

from util.mcmc_utils import *
#from util.mcmc_transitions import *
import numpy as np
import matplotlib.pyplot as plt

# Check if running on a headless system

import tractor
print "Tractor loaded from: " + tractor.__file__
import astrometry
print "Astrometry loaded from: " + astrometry.__file__

print "Loading images..."

with Timing("loading"):
  multitractor = loadTractorSingleImage(bands=['r','i','g'], catalog=True)

print "Saving true images..."
fci = falseColorImage(multitractor, useModel=False)
fci.save("fci_truth.png")
fci = falseColorImage(multitractor, useModel=True)
fci.save("fci_catalog.png")

print "Initializing and saving initialization image..."

initializeTractor(multitractor, threshold=1000)
fci = falseColorImage(multitractor, useModel=True)
fci.save("fci_init.png")

rand = np.random.RandomState(seed=100)

from collections import defaultdict
b = defaultdict(lambda: defaultdict(list))
lls = list()

def cb(tractor, it, logprob, _memo_unused):
    for isrc, src in enumerate(tractorBrightnesses(tractor)):
        for band, val in src.items(): 
            b[isrc][band].append(val)
    lls.append(logprob)

doMCMC(multitractor, iters=10, rand=rand,
       aPrior=1./3, bPrior=1e-4, eta=1e-1,
       sliceW=3e-5, sliceM=20,
       allowBirthDeath=True,
       allowMergeSplit=True,
       cb=cb)

print "==== RESULTS ===="

print tractorBrightnesses(multitractor)

plt.figure()
plt.plot(b[0]['i'], 'r')
plt.plot(b[0]['r'], 'g')
plt.plot(b[0]['g'], 'b')
plt.plot(b[1]['i'], 'r--')
plt.plot(b[1]['r'], 'g--')
plt.plot(b[1]['g'], 'b--')
plt.plot(b[2]['i'], 'r-.')
plt.plot(b[2]['r'], 'g-.')
plt.plot(b[2]['g'], 'b-.')
plt.xlabel("MCMC iteration after source's birth")
plt.ylabel("Band-wise brightnesses (counts)")
plt.savefig("brightnesses.pdf")

print "Log likelihoods:", lls

plt.figure()
plt.plot(lls)
plt.xlabel("MCMC iteration")
plt.ylabel("Log likelihood")
plt.savefig("lls.pdf")

fci = falseColorImage(multitractor, useModel=True)
fci.save("fci_iters.png")
