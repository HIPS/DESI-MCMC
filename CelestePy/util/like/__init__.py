from like_list import *
from gmm_like import *

# try to import cython gmm - otherwise default to einsum implementation
try:
    from gmm_like_fast import gmm_like_2d as gmm_like_2d_cy
    def gmm_like_2d(x, ws, mus, sigs, probs=None):
        if probs is None:
            probs = np.zeros(x.shape[0], dtype=np.float)  # bufer for prob values
        gmm_like_2d_cy(probs, x, ws, mus, sigs)
        return probs

except:
    gmm_like_2d = gmm_prob

