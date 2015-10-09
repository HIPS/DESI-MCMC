import os
import numpy as np
from scipy.stats import norm
import random

from get_spec import simulated_annealing

"""
Generates data from normal distribution with known variance and mean drawn from
uniform. We then run AIS using prior of the correct prior vs. a different
uniform prior. The correct one should have a higher ccalculated marginal
likelihood.
"""

NORMAL_SD = 2
UNIF_UPPER = 20
UNIF_LOWER = 0

NUM_VALUES = 10
def generate_normal_data():
    mu = random.random() * (UNIF_UPPER - UNIF_LOWER) + UNIF_LOWER
    print "generated from", mu
    return norm.rvs(mu, NORMAL_SD, NUM_VALUES)

def normal_sample_prior():
    return np.array([random.random() * (UNIF_UPPER - UNIF_LOWER) + UNIF_LOWER])

def normal_sample_double_prior():
    return np.array([random.random() * (2 * UNIF_UPPER - UNIF_LOWER) + UNIF_LOWER])

def normal_prior_logpdf(values):
    mu = values[0]
    if mu < UNIF_LOWER or mu > UNIF_UPPER:
        return -np.inf
    return -np.log(UNIF_UPPER - UNIF_LOWER)

def normal_double_prior_logpdf(values):
    mu = values[0]
    if mu < UNIF_LOWER or mu > 2 * UNIF_UPPER:
        return -np.inf
    return -np.log(2 * UNIF_UPPER - UNIF_LOWER)

def normal_likelihood_logpdf(values, data):
    mu = values[0]
    likelihood = np.sum([norm.logpdf(datum, mu, NORMAL_SD) for datum in data])
    return likelihood

def normal_posterior_logpdf(values, data):
    return normal_likelihood_logpdf(values, data) + normal_prior_logpdf(values)

def normal_double_posterior_logpdf(values, data):
    return normal_likelihood_logpdf(values, data) + normal_double_prior_logpdf(values)

PROPOSE_SD = 1.
def propose_normal(values, log_likelihood):
    mu = values[0]
    trials = 1
    for i in range(trials):
        mu_new = norm.rvs(mu, PROPOSE_SD)
        try:
            prob_new = log_likelihood(np.array([mu_new]))
            prob_old = log_likelihood(np.array([mu]))
        except:
            continue

        if prob_new == -np.inf or prob_old == -np.inf:
            continue
        if prob_new > prob_old or random.random() < np.exp(prob_new - prob_old):
            mu = mu_new

    return np.array([mu])


if __name__ == "__main__":
    xs = generate_normal_data()
    ell1 = simulated_annealing(normal_sample_prior,
                               normal_prior_logpdf,
                               normal_posterior_logpdf,
                               propose_normal,
                               xs);

    print "Correct one:", ell1
    """
    ell2 = simulated_annealing(normal_sample_double_prior,
                               normal_double_prior_logpdf,
                               normal_double_posterior_logpdf,
                               propose_normal,
                               xs);

    print "Wrong one:", ell2
"""
