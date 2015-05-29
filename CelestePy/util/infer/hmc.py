"""

Implementation of Hybrid Monte Carlo (HMC) sampling algorithm following Neal (2010).
Use the log probability and the gradient of the log prob to navigate the distribution.

Scott Linderman <slinderman@seas.harvard.edu>
2012-2014

(Update 5/14/2015)
Andrew Miller <acm@seas.harvard.edu>

"""
import numpy as np
import numpy.random as npr

def hmc(x_curr,
        llhfunc,
        grad_llhfunc,
        eps,
        num_steps,
        mass                  = None,
        num_iter              = 1,
        p_curr                = None,
        refresh_alpha         = 0.0,
        adaptive_step_sz      = False,
        tgt_accept_rate       = 0.9,
        avg_accept_time_const = 0.95,
        avg_accept_rate       = 0.9,
        min_step_sz           = 0.00005,
        max_step_sz           = 1.0,
        negative_log_prob     = True):
    """
    U       - function handle to compute log probability we are sampling
    grad_U  - function handle to compute the gradient of the density with respect 
              to relevant params
    step_sz - step size
    n_steps - number of steps to take
    q_curr  - current state

    negative_log_prob   - If True, assume U is the negative log prob

    """
    new_accept_rate = 0.
    imass = 1./mass
    def energy(X): 
        return -llhfunc(X)

    def grad_energy(X):
        return -grad_llhfunc(X)

    def hamiltonian(X, P):
        return energy(X) + .5*np.sum(imass*P*P) 

    # define leapfrog step (or multiple steps)
    def leapstep(xx0, pp0):
        xx, pp = xx0.copy(), pp0.copy()
        pph    = pp - .5 * eps * grad_energy(xx)  # half step first step
        for l in xrange(num_steps):
            xx      = xx + eps * imass * pph
            eps_mom = .5*eps if l==num_steps-1 else eps    # half step on last jump
            pph     = pph - eps_mom*grad_energy(xx)
        return xx, pph

    # sample initial momentum
    X = x_curr.copy()
    if p_curr is None:
        P = np.sqrt(mass)*npr.randn(X.shape[0])
    else:
        P = p_curr.copy()
    ll_curr = -hamiltonian(X, P)
    for i in xrange(num_iter):
        # (partial) refresh momentum
        P = refresh_alpha*P + np.sqrt(1.0 - refresh_alpha**2)*np.sqrt(mass)*npr.randn(X.shape[0])
        Xp, Pp = leapstep(X, P)
        Pp     = -Pp

        ll_prop = -hamiltonian(Xp, Pp)
        accept = np.log(npr.rand()) < ll_prop - ll_curr
        if accept:
            X       = Xp
            P       = Pp
            ll_curr = ll_prop

        # re-negate the momentum regardless of accept/reject
        P = -P

        # Do adaptive step size updates if requested
        if adaptive_step_sz:
            new_accept_rate = avg_accept_time_const * avg_accept_rate + \
                              (1.0-avg_accept_time_const) * accept
            if avg_accept_rate > tgt_accept_rate:
                eps = eps * 1.02
            else:
                eps = eps * 0.98
            eps = np.clip(eps, min_step_sz, max_step_sz)

    # return X, P, some other info if not adaptive
    return X, P, eps, new_accept_rate


def test_hmc():
    """
    Test HMC on a Gaussian distribution
    """
    from scipy.stats import norm
    mu = 0
    sig = 1
    p = norm(mu, sig).pdf

    f = lambda x: -0.5*x**2
    grad_f = lambda x: -x

    N_samples = 10000
    smpls = np.zeros(N_samples)
    for s in np.arange(1,N_samples):
        smpls[s] = hmc(lambda x: -1.0*f(x),
                       lambda x: -1.0*grad_f(x),
                       0.1, 10,
                       np.atleast_1d(smpls[s-1]),
                       negative_log_prob=True)
    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(smpls, 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p(bincenters), 'r--', linewidth=1)
    plt.show()

def test_gamma_linear_regression_hmc():
    """
    Test ARS on a gamma distributed coefficient for a gaussian noise model
    y = c*x + N(0,1)
    c ~ gamma(2,2)
    """
    a = 6.
    b = 1.
    x = 1
    sig = 1.0
    avg_accept_rate = 0.9
    stepsz = 0.01
    nsteps = 10
    N_samples = 10000

    from scipy.stats import gamma, norm
    g = gamma(a, scale=1./b)
    prior = lambda logc: a * logc -b*np.exp(logc)
    dprior = lambda logc: a -b*np.exp(logc)
    lkhd = lambda logc,y: -0.5/sig**2 * (y-np.exp(logc)*x)**2
    dlkhd = lambda logc,y: 1.0/sig**2 * (y-np.exp(logc)*x) * np.exp(logc)*x
    posterior = lambda logc,y: prior(logc) + lkhd(logc,y)
    dposterior = lambda logc,y: dprior(logc) + dlkhd(logc,y)

    logc_smpls = np.zeros(N_samples)
    y_smpls = np.zeros(N_samples)
    logc_smpls[0] = np.log(g.rvs(1))
    y_smpls[0] = np.exp(logc_smpls[0]*x) + sig*np.random.randn()

    for s in np.arange(1,N_samples):
        if np.mod(s, 100) == 0:
            print "Sample ", s
        # Sample y given c
        y_smpls[s] = np.exp(logc_smpls[s-1])*x + sig*np.random.randn()

        # Sample c given y
        logc_smpls[s], stepsz, avg_accept_rate =  \
            hmc(lambda logc: -1.0*posterior(logc, y_smpls[s]),
                lambda logc: -1.0*dposterior(logc, y_smpls[s]),
                stepsz, nsteps,
                logc_smpls[s-1].reshape((1,)),
                avg_accept_rate=avg_accept_rate,
                adaptive_step_sz=True)

    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(np.exp(logc_smpls), 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, g.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

if __name__ == '__main__':
    test_hmc()
    # test_gamma_linear_regression_hmc()
