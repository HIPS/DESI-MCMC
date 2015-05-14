import numpy as np

"""
Grab bag of misc functions

 - check gradient of a scalar valued function
 - ParamParser object, for parsing vectors with names (replace with numpy structured array)
"""

def check_grad(fun, jac, th, compwise=False):
    """ check the gradient along a random direction """
    if compwise:
        grad_th = np.zeros(th.shape)
        for e in xrange(len(th)):
            de = np.zeros(th.shape)
            de[e] = 1e-5
            grad_th[e] = (fun(th+de) - fun(th-de))/2e-5

        grad_jac = jac(th)
        print "component-wise diffs = %s"%np.str(grad_jac - grad_th)
        print "comp-wise relative diffs = %s"%np.str((grad_jac - grad_th) / np.abs(grad_th))

    else: #check along random direction
        param_scale = .1
        rand_dir    = np.random.randn(th.size) * param_scale
        rand_dir    = rand_dir / np.sqrt(np.dot(rand_dir, rand_dir))
        test_fun    = lambda x : fun(th + x * rand_dir.reshape(th.shape))
        nd          = (test_fun(1e-4) - test_fun(-1e-4)) / 2e-4
        ad          = np.dot(jac(th).ravel(), rand_dir)
        print "Random dir check: Relative diff is: {0}".format((nd - ad)/np.abs(nd))


class ParamParser(object):
    """ Helper class to handle different slicing for different parameters
    in one long vector """
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def set(self, vect, name, val):
        idxs, shape = self.idxs_and_shapes[name]
        vect[idxs] = val.ravel()

    def get_slice(name):
        return self.idxs_and_shapes[name][0]


