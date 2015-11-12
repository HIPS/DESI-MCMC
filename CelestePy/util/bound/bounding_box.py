import scipy.stats
import numpy as np

"""
Based on writeup bounding_box.pdf. Finds and returns a radius R such that a
circle with radius R around (0, 0) contains (1 - error) of the probability
mass of the mixture of Gaussians defined by weights, means, covars.
"""
def calc_bounding_radius(weights, means, covars, error, center=np.array([0, 0])):
    minbound = -np.inf
    # get quantile for chi square
    Rsq = scipy.stats.chi2.ppf(1 - error, 2)
    for i in range(len(weights)):
        sigma1 = np.sqrt(covars[i, 0, 0])
        sigma2 = np.sqrt(covars[i, 1, 1])
        rho = covars[i, 0, 1] / (sigma1 * sigma2)

        A11 = sigma1
        A21 = rho * sigma2
        A22 = sigma2 * np.sqrt(1 - rho**2.)

        An = 1. / Rsq * (1 / A11**2. + A21**2. / A22**2.)
        Bn = 1. / Rsq * (-2 * A21 / (A11 * A22**2.))
        Cn = 1. / Rsq * 1. / A22**2.

        majaxis = (0.5 * (An + Cn - np.sqrt(Bn**2. + (An - Cn)**2.)))**(-0.5)
        dist = np.sqrt((means[i, 0] - center[0])**2. + (means[i, 1] - center[1])**2.)

        minbound = max(minbound, majaxis + dist)

    return minbound

