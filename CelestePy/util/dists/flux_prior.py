from mog import MixtureOfGaussians
import autograd.numpy as np

class FluxColorMoG(MixtureOfGaussians):
    """ prior distribution for flux colors """

    A = np.zeros((5,5))
    A[0,0] = A[1,1] = A[3,2] = A[4,3] = 1.
    A[2, :]  = -1
    A[2, -1] = 1.
    Ainv = np.linalg.inv(A)

    def __init__(self, means, covs, pis):
        super(FluxColorMoG, self).__init__(means, covs, pis)
        # create transformation matrix that takes log [u g r i z] fluxes, 
        # and turns them into [lu - lr, lg - lr, li - lr, lz - lr, lr]
        #
        # the mixture of gaussians is then a law on this transformed 
        # space
        #
        #  [u g r i z] dot [ 1   0   0   0   0
        #                    0   1   0   0   0
        #                   -1  -1  -1  -1   1
        #                    0   0   1   0   0
        #                    0   0   0   1   0
        #
        #self.A = A
        #self.Ainv = np.linalg.inv(A)

    @staticmethod
    def to_colors(x):
        return np.dot(x, FluxColorMoG.A)

    @staticmethod
    def to_fluxes(colors):
        return np.dot(colors, FluxColorMoG.Ainv)

    def logpdf(self, x):
        return super(FluxColorMoG, self).logpdf(np.dot(x, self.A))


class GalShapeMoG(MixtureOfGaussians):
    """ prior distribution for flux colors """

    def __init__(self, means, covs, pis):
        super(GalShapeMoG, self).__init__(means, covs, pis)

    @staticmethod
    def to_unconstrained(x):
        sigma, ab, phi = x[0], x[1], x[2]
        return np.array([ np.log(sigma),
                          np.log(ab) - np.log(1.-ab),
                          np.log(phi) - np.log(np.pi - phi) ])

    def to_constrained(x):
        raise NotImplementedError

    def logpdf(self, x):
        return super(GalShapeMoG, self).logpdf(self.to_unconstrained(x))


class GalRadiusMoG(MixtureOfGaussians):
    """ prior distribution for flux colors """

    def __init__(self, means, covs, pis):
        super(GalRadiusMoG, self).__init__(means, covs, pis)

    @staticmethod
    def to_unconstrained(x):
        return np.log(x)

    def to_constrained(x):
        raise NotImplementedError

    def logpdf(self, x):
        return super(GalRadiusMoG, self).logpdf(self.to_unconstrained(x))


class GalAbMoG(MixtureOfGaussians):
    """ prior distribution for flux colors """

    def __init__(self, means, covs, pis):
        super(GalAbMoG, self).__init__(means, covs, pis)

    @staticmethod
    def to_unconstrained(x):
        return np.log(x) - np.log(1.-x)

    def to_constrained(x):
        raise NotImplementedError

    def logpdf(self, x):
        return super(GalAbMoG, self).logpdf(self.to_unconstrained(x))

