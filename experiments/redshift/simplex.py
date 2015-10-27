"""
methods for manipulating vectors on the simplex.

Includes methods to generate multinomial logistic params on S^D+1 from
real valued vectors in R^D (and back)
"""
import autograd.numpy        as np
import autograd.numpy.random as npr

def logit(p):
    """takes rows of R^D+1 vectors on the simplex and outputs R^D logit values.
    Input:
        p: N x D+1 non-negative matrix such that each row sums to 1
    Output:
        x: N x D matrix of real valued such that the softmax of x yields p
    Note: this is the inverse transformation of logistic
    """
    x = np.log(p) - np.log(p[:,-1,np.newaxis])
    x -= x[:,-1,np.newaxis]
    return x[:,:-1]


def logistic(x):
    """takes rows R^D vectors in general position and outputs R^D+1 vectors
    on the simplex.
    Input:
        x: aN x D+1 non-negative matrix such that each row sums to 1
    Output:
        p: N x D matrix of real valued such that the softmax of x yields p
    Note: this is the inverse transformation of logit
    """
    x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    x -= np.max(x, axis=1)[:,np.newaxis] # subtract off to prevent overflow
    p = np.exp(x)
    return p / np.sum(p, axis=1)[:,np.newaxis]


def rand_logistic_norm(N, K):
    """ Randomly sample N rows, each of K dimensions such that each row sums
    to one"""
    z = npr.randn(N, K-1)
    return logistic(z)


def test_logit():
    """ simple test to ensure logistic(logit(x)) = x """
    p = rand_logistic_norm(10, 4)
    x = logit(p)
    pt = logistic(x)
    assert np.allclose(p, pt), "Test logit fails!"
    print "test_logit passes!"

if __name__=="__main__":
    test_logit()
