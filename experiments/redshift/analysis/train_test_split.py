import fitsio
import numpy as np
import seaborn as sns

dr10file = fitsio.FITS('../../data/DR10QSO/DR10Q_v2.fits')
qso_data = np.column_stack((dr10file[1]['PSFFLUX'].read(),
                            dr10file[1]['Z_VI'].read()))

##
## split randomly (half and half)
##
np.random.seed(120)
rperm = np.random.permutation(qso_data.shape[0])
train_idx = rperm[:len(rperm)/2]
test_idx = rperm[len(rperm)/2:]

f = file("split_random.bin","wb")
np.save(f,train_idx)
np.save(f,test_idx)
f.close()

plt.figure(1)
n, zbins, patches = plt.hist(qso_data[train_idx, -1], 40, alpha=.25, color='blue', label='train', normed=True)
n, zbins, patches = plt.hist(qso_data[test_idx, -1], zbins, alpha=.25, color='red', label='test', normed=True)

plt.figure(2)
n, binsr, patches = plt.hist(qso_data[train_idx,2], 40, alpha=.25, color='blue', label='train', normed=True)
n, binsr, patches = plt.hist(qso_data[train_idx,2], binsr, alpha=.25, color='red', label='test', normed=True)
plt.show()


##
## split by selectively - train on low z, test on hi z
##

# get 75th percentile
#zorder = np.argsort(qso_data[:,-1])
z_cutoff = np.percentile(qso_data[:,-1], 85)
train_idx = np.where(qso_data[:,-1] <= z_cutoff)[0]
test_idx  = np.where(qso_data[:,-1] > z_cutoff)[0]

f = file("split_redshift.bin","wb")
np.save(f,train_idx)
np.save(f,test_idx)
f.close()

plt.figure(1)
n, bins, patches = plt.hist(qso_data[train_idx, -1], zbins, alpha=.25, color='blue', label='train', normed=True)
n, bins, patches = plt.hist(qso_data[test_idx, -1], zbins, alpha=.25, color='red', label='test', normed=True)
plt.legend()

plt.figure(2)
n, binsr, patches = plt.hist(qso_data[train_idx,2], binsr, alpha=.25, color='blue', label='train', normed=True)
n, binsr, patches = plt.hist(qso_data[train_idx,2], binsr, alpha=.25, color='red', label='test', normed=True)
plt.legend()
plt.show()


##
## split selectively by low R band and hi R band
##
r_cutoff = np.percentile(qso_data[:,2], 10)
train_idx = np.where(qso_data[:,2] > r_cutoff)[0]
test_idx  = np.where(qso_data[:,2] <= r_cutoff)[0]

f = file("split_flux.bin","wb")
np.save(f,train_idx)
np.save(f,test_idx)
f.close()

plt.figure(1)
n, bins, patches = plt.hist(qso_data[train_idx, -1], zbins, alpha=.25, color='blue', label='train', normed=True)
n, bins, patches = plt.hist(qso_data[test_idx, -1], zbins, alpha=.25, color='red', label='test', normed=True)
plt.legend()

plt.figure(2)
n, bins, patches = plt.hist(qso_data[train_idx,2], binsr, alpha=.25, color='blue', label='train', normed=True)
n, bins, patches = plt.hist(qso_data[train_idx,2], binsr, alpha=.25, color='red', label='test', normed=True)
plt.legend()
plt.show()




