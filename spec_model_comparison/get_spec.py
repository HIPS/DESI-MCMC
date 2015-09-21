import fitsio
import os
import numpy as np
from scipy.stats import invgamma, dirichlet, multivariate_normal, norm
import urllib
import random

# Bayesian model selection scheme
#
# Let's say we have model M. Then, the model evidence is:
#     P(M | D) \propto P(D | M) P(M)
# Typically, we will assume P(M) is uniform across the finite number of models.
# P(D | M) is called the marginal likelihood and is:
#     \int P(D | M, theta) P(theta | M) dtheta
# We use importance simulated annealing

TRIALS = 2

def inverse_gamma_pdf(x, a, loc, scale):
    return invgamma.pdf(x, a, loc=loc, scale=scale)

def uniform_pdf(x, start, end):
    return 1 / (end - start)

# weights, means, scales
COV_ALPHA = 5

# define uniform interval
START = 0
END = 2000

A = 1
LOC = 10
SCALE = 10
components = 10
def simple_model_sample_prior():
    vals = np.zeros(3 * components)
    vals[:components] = \
        np.exp(multivariate_normal.rvs(cov=COV_ALPHA*np.eye(components)))
    for i in range(components, 2*components):
        vals[i] = random.random() * (END - START) + START

    vals[(2*components):(3*components)] = \
        invgamma.rvs(A, LOC, SCALE, size=components)

    return vals

def simple_model_prior_pdf(values):
    means = np.prod([uniform_pdf(val, START, END) for val in values[components:(2*components)]])
    scales = np.prod([invgamma.pdf(val, A, LOC, SCALE) for val in values[(2*components):(3*components)]])
    weights = multivariate_normal.pdf(values[:components],
                                      mean=np.zeros(components),
                                      cov=COV_ALPHA*np.eye(components))
    return weights * means * scales

def simple_model_posterior_pdf(values, data):
    prob = 1
    for datum in data:
        lambdas = datum['lam']
        spec    = datum['flux']
        ivar    = datum['ivar']
        for i in range(len(lambdas)):
            pred = 0
            for n in range(components):
                weight = values[n]
                mean = values[components + n]
                scale = values[2 * components + n]

                pred += np.exp(-(lambdas[i] - mean)**2 / scale) * weight

            prob *= norm.pdf(spec[i], pred, 1 / np.sqrt(ivar[i]))

    return prob * simple_model_prior_pdf(values)

MEAN_VAR = 10
VAR_VAR = 10
WEIGHT_VAR = 1
def propose(values, likelihood):
    trials = 1
    curr_values = np.copy(values)
    for i in range(trials):
        # propose weights
        log_weights = np.log(values[:components])
        new_log_weights = multivariate_normal.rvs(mean=log_weights, cov=np.eye(components)*WEIGHT_VAR)
        # convert from softmax
        new_weights = np.exp(new_log_weights)
    
        # propose means
        means = values[components:(2*components)]
        new_means = multivariate_normal.rvs(mean=means, cov=np.eye(components)*MEAN_VAR)

        # propose variances
        variances = values[(2*components):(3*components)]
        new_vars = multivariate_normal.rvs(mean=variances, cov=np.eye(components)*VAR_VAR)

        new_value = np.append(new_weights, np.append(new_means, new_vars))
        prob_new = likelihood(new_value)
        prob_old = likelihood(values)
        if np.any(means < 0) or np.any(variances < 0) or \
                prob_new > prob_old or random.random() < prob_new / prob_old:
            curr_values = np.copy(new_value)

    return curr_values


# define betas
# define steps in simulated annealing
# proposal variance
# dimensions
betas = np.append(np.linspace(0,0.1,num=10), \
        np.append(np.linspace(0.1,1,num=10), np.linspace(1,100,10)))

def get_intermediate(prior, posterior, x, data, beta):
    return prior_pdf(x)**(1 - beta) * posterior_pdf(x, data)**beta

def simulated_annealing(sample_prior, prior_pdf, posterior_pdf, data):
    sum_weights = 0
    weighted_sum = 0
    for i in range(TRIALS):
        values = np.zeros((len(betas), 3 * components))
        likelihoods = np.zeros(TRIALS)

        # sample from prior
        values[0,:] = sample_prior()

        for j in range(1, len(betas)):
            # Metropolis step
            values[j,:] = \
                propose(values[j-1,:],
                        lambda x:(prior_pdf(x)**(1 - betas[j]) * posterior_pdf(x, data)**betas[j]))

        weight = 1
        for j in range(1, len(betas)):
            weight *= \
                prior_pdf(values[j,:])**(betas[j-1] - betas[j]) * \
                posterior_pdf(values[j,:], data)**(betas[j] - betas[j-1])

        sum_weights += weight
        weighted_sum += weight * posterior_pdf(values[len(betas)-1,:], data)

    return weighted_sum / sum_weights

#
# Constructs a path to the spec file online given plate, mjd, fiber - 
# which is information about the spec instrument/run
#

def spec_url(plate, mjd, fiber):
    spec_url_template = os.path.join(
        "http://data.sdss3.org/sas/dr12/boss/spectro/",
        "redux/v5_7_0/spectra/%04d/spec-%04d-%05d-%04d.fits"
    )
    return spec_url_template%(plate, plate, mjd, fiber)

if __name__=="__main__":

    # load source file - has information about spec collection
    df = fitsio.FITS('PhotoSpecBoss_andrewcmiller.fit')
    sources = df[1].read()

    # easily identify which objects are which
    gal_idx = sources['class'] == "GALAXY"
    qso_idx = sources['class'] == "QSO"
    str_idx = sources['class'] == "STAR"
    print "%d galaxies"%sum(gal_idx)

    # grab the URL for the first star
    s = sources[str_idx][0]
    first_star = spec_url(s['plate'], s['mjd'], s['fiberID'])
    print first_star

    f = urllib.urlopen(first_star)
    out = open('spec.fits', 'wb')
    out.write(f.read())
    out.close()

    df = fitsio.FITS('spec.fits')
    spec_info = {}
    spec_info['lam'] = np.power(10., df[1]['loglam'].read())
    spec_info['flux'] = df[1]['flux'].read()
    spec_info['ivar'] = df[1]['ivar'].read()
    print spec_info

    # get actual spectrum 
    simulated_annealing(simple_model_sample_prior,
                        simple_model_prior_pdf,
                        simple_model_posterior_pdf,
                        [spec_info])

