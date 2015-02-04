import numpy as np
from quasar_fit_basis import load_basis_fit
from sklearn.mixture.gmm import GMM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

basis_cache = 'cache/basis_fit_K-4_V-1364.pkl'
th, lam0, lam0_delta, parser = load_basis_fit(basis_cache)

# compute actual weights and basis values (normalized basis + weights)
mus    = parser.get(th, 'mus')
betas  = parser.get(th, 'betas')
omegas = parser.get(th, 'omegas')
# print omegas
W = np.exp(omegas)
W = W / np.sum(W, axis=1,keepdims=True)
# print W

for i in range(omegas.shape[0]):
    omegas[i,0] = 0
    for j in range(1, omegas.shape[1]):
        omegas[i,j] = np.log(W[i,j]) - np.log(W[i,0])
#print omegas

W = np.exp(omegas)
W = W / np.sum(W, axis=1, keepdims=True)

# print W

# plot projection
pca = PCA(n_components=2)
pca.fit(omegas[:,1:4])
projs = pca.transform(omegas[:,1:4])
plt.plot(projs[:,0], projs[:,1], 'bo')
plt.savefig('projs.png')
plt.close()

# fit mixture of gaussians
N = omegas.shape[0]
for k in range(1,10): 
    g = GMM(n_components=k, covariance_type='full')
    mixture = g.fit(omegas[0:(int(0.75*N)),1:4])
    print k, ":", sum(mixture.score_samples(omegas[(int(0.75*N)):N,1:4])[0])

g = GMM(n_components=3, covariance_type='full')
mixture = g.fit(omegas[:,1:4])
print "weights:", mixture.weights_
print "means:", mixture.means_
print "covariances:", mixture.covars_

A = np.transpose(pca.transform(np.eye(3)))
print A

new_means = np.zeros((3, 2))
new_covars = np.zeros((3, 2, 2))

print mixture.means_.shape
for k in range(3):
    new_means[k,:] = np.dot(A, mixture.means_[k])
    new_covars[k,:,:] = np.dot(np.dot(A, mixture.covars_[k]), np.transpose(A))

print new_means
print new_covars

delta = 0.025 
xs = np.arange(-0.7, 0.5, delta)
ys = np.arange(-4.5, 4.0, delta)
probs = np.zeros((len(xs), len(ys)))

for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        for k in range(3):
            probs[i][j] += \
                mixture.weights_[k] * \
                multivariate_normal.pdf((x,y), mean=new_means[k], cov=new_covars[k])

X, Y = np.meshgrid(xs, ys)
CS = plt.contour(X, Y, np.transpose(probs), 10)
plt.clabel(CS, inline=3, fontsize=10)
plt.savefig('probs_contour.png')

