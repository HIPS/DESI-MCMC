import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import scipy.misc as scpm
from CelestePy.util.infer.slicesample import slicesample

def parallel_temper_slice(lnpdf, x0, Nsamps, Nchains, temps=None, 
                          callback=None, verbose=True, printskip=20, 
                          compwise = True):

    # verify temperature schedule
    if temps is None:
        temps      = np.linspace(.2, 1., Nchains)
    assert len(temps) == Nchains, "Nchains must be = len(temps)!"
    assert temps[-1] == 1., "temps[-1] must be = 1. or else you're not doing it right..."

    # keep track of swaps into every temperature state
    Nswaps = np.zeros(len(temps)) 
    print " === parallel tempering ==="
    print "  with temps ", np.str(temps)

    # set up printing
    def printif(string, condition):
        if condition:
            print string
    printif("{iter:10}|{ll:10}|{num_swaps:10}|{cold_swaps:12}|{th0:10}".format(
                iter       = " iter ",
                ll         = " ln_post ", 
                num_swaps  = " Nswaps ",
                cold_swaps = " NColdSwaps ",
                th0        = " th0 (mean, sd)"), verbose)

    # set up sample array
    assert x0.shape[0] == Nchains, "initial x has to have shape Nchains x Dim"
    D = x0.shape[1]
    chain        = np.zeros((Nchains, Nsamps, D))
    chain[:,0,:] = x0.copy()
    chain_lls    = np.zeros((Nchains, Nsamps))
    for ci in xrange(Nchains):
        chain_lls[ci, 0] = temps[ci] * lnpdf(chain[ci, 0, :])

    # draw samples
    for s in np.arange(1, Nsamps):

        # Nchains HMC draws
        for ci in range(Nchains):
            chain[ci, s, :], chain_lls[ci, s] = slicesample(
                    init_x   = chain[ci][s-1,:],
                    logprob  = lambda(x): temps[ci] * lnpdf(x),
                    compwise = compwise
                )

        # propose swaps cascading down from first 
        for ci in range(Nchains-1):
            # cache raw ll's for each (already computed)
            ll_ci      = chain_lls[ci][s] / temps[ci]
            ll_ci_plus = chain_lls[ci+1][s] / temps[ci + 1]

            # propose swap between chain index ci and ci + 1
            ll_prop = temps[ci]*ll_ci_plus + temps[ci+1]*ll_ci
            ll_curr = chain_lls[ci][s] + chain_lls[ci+1][s]
            if np.log(npr.rand()) < ll_prop - ll_curr:
                ci_samp = chain[ci, s, :].copy()

                # move chain sample ci+1 into ci
                chain[ci, s, :]   = chain[ci+1, s, :]
                chain_lls[ci, s]  = temps[ci] * ll_ci_plus

                # move chain sample ci into ci + 1
                chain[ci+1, s, :]  = ci_samp
                chain_lls[ci+1, s] = temps[ci+1]*ll_ci

                # track number of swaps
                Nswaps[ci+1] += 1

        printif("{iter:10}|{ll:10}|{num_swaps:10}|{cold_swaps:12}|{th0:10}".format(
                iter       = "%d/%d"%(s, Nsamps),
                ll         = " %2.4g "%chain_lls[-1, s],
                num_swaps  = " %d "%np.sum(Nswaps),
                cold_swaps = " %d"%Nswaps[-1],
                th0        = " %2.2f (%2.2f, %2.2f)"%(chain[-1,s,0], chain[-1,:s,0].mean(), chain[-1,:s,0].std())), 
                verbose and s%printskip==0)

        if callback is not None:
            callback(s, chain, chain_lls)

    #only return the chain we care about
    return chain, chain_lls


if __name__=="__main__":

    # Create a random parameterization.
    def gen_mog_2d(seed=105, K=3, alpha=1.):
        rng   = npr.RandomState(seed)
        means = rng.randn(K,2)
        covs  = rng.randn(K,2,2)
        covs  = np.einsum('...ij,...kj->...ik', covs, covs)
        icovs = np.array([npla.inv(cov) for cov in covs])
        dets  = np.array([npla.det(cov) for cov in covs])
        chols = np.array([npla.cholesky(cov) for cov in covs])
        pis   = rng.dirichlet(alpha*np.ones(K))
        return means, covs, icovs, dets, chols, pis
    means, covs, icovs, dets, chols, pis = gen_mog_2d()
    mu = np.sum(pis * means.T, axis=1)

    # mog log like
    def mog_loglike(x, means, icovs, dets, pis):
        xx = np.atleast_2d(x)
        centered = xx[:,:,np.newaxis] - means.T[np.newaxis,:,:]
        solved   = np.einsum('ijk,lji->lki', icovs, centered)
        logprobs = -0.5*np.sum(solved * centered, axis=1) - np.log(2*np.pi) - 0.5*np.log(dets) + np.log(pis)
        logprob  = scpm.logsumexp(logprobs, axis=1)
        if len(x.shape) == 1:
            return logprob[0]
        else:
            return logprob

    # mog perfect sampler
    def mog_samples(N, means, chols, pis):
        def discrete(p, shape):
            length = np.prod(shape)
            indices = p.shape[0] - np.sum(npr.rand(length)[:,np.newaxis] < np.cumsum(p), axis=1)
            return indices.reshape(shape)
        K, D = means.shape
        indices = discrete(pis, (N,))
        n_means = means[indices,:]
        n_chols = chols[indices,:,:]
        white   = npr.randn(N,D)
        color   = np.einsum('ikj,ij->ik', n_chols, white)
        return color + n_means

    #test on simple two d example
    Nchains = 5
    Nsamps  = 10000
    x0      = np.random.randn(Nchains, 2)
    chain, chain_lls = parallel_temper_slice(
        lnpdf   = lambda(x): mog_loglike(x, means, icovs, dets, pis),
        x0      = np.random.randn(Nchains, 2),
        Nsamps  = Nsamps,
        Nchains = Nchains,
        verbose = True, printskip=100)

    # perfect samples for comparison
    S = mog_samples(Nsamps, means, chols, pis)

    # diagnostics
    print "true mean: ", np.str(mu)
    print "samp mean: ", np.str(chain[-1,:,:].mean(axis=0))

    import matplotlib.pyplot as plt
    Sm = np.cumsum(S, axis=0) / (np.arange(Nsamps)[:,np.newaxis]+1.0)
    Xm = np.cumsum(chain[-1,:,:], axis=0) / (np.arange(Nsamps)[:,np.newaxis]+1.0)
    plt.figure(1)
    plt.plot(np.arange(Nsamps), Sm, 'k-',
             np.arange(Nsamps), Xm, 'b-')
    plt.show()






