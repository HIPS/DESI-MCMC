import numpy as np
import numpy.random as npr

def slicesample_step_out(xx, llh_func, last_llh=None, step=1, step_out=True, x_l=None, x_r=None, lb=-np.Inf, ub=np.Inf):
    """ Generic Slice Sampler: randomly permutes dimensions and does 
        slice sampling with stepping out

        Input: 
            - xx       : initial state (numpy array with shape[0] = Dimension)
            - llh_func : log likelihood function - llh_func(xx)
            - last_llh : log likelihood of sample xx, if cached
            - sigma    : stepping out values

        Output:
            - xx_samp  : new sample leaving llh_func invariant
            - llh      : log likelihood of xx_samp
    """
    xx = np.atleast_1d(xx)
    dims = xx.shape[0]
    perm = range(dims)
    np.random.shuffle(perm)

    assert np.all(xx < ub), "xx (%s) >= ub (%s)"%(np.array_str(xx), np.array_str(ub))
    assert np.all(xx > lb), "xx (%s) <= lb (%s)"%(np.array_str(xx), np.array_str(lb))

    if isinstance(step, int) or isinstance(step, float) or \
        isinstance(step, np.int) or isinstance(step, np.float):
        step = np.tile(step, dims)
    elif isinstance(step, tuple) or isinstance(step, list):
        step = np.array(step)
 
    if last_llh is None:
        last_llh = llh_func(xx)
 
    for d in perm:
        llh0 = last_llh + np.log(np.random.rand())
        rr = np.random.rand()
        if x_l is None:
            x_l    = xx.copy()
            x_l[d] = max(x_l[d] - rr*step[d], lb[d])
        else:
            x_l = np.atleast_1d(x_l)
            assert x_l.shape == xx.shape
            assert np.all(x_l <= xx)
        if x_r is None:
            x_r    = xx.copy()
            x_r[d] = min(x_r[d] + (1-rr)*step[d], ub[d])
        else:
            x_r = np.atleast_1d(x_r)
            assert x_r.shape == xx.shape
            assert np.all(x_r >= xx)

        if step_out:
            llh_l = llh_func(x_l)
            while llh_l > llh0 and x_l[d] > lb[d]:
                x_l[d] = max(x_l[d] - step[d], lb[d])
                llh_l  = llh_func(x_l)
            llh_r = llh_func(x_r)
            while llh_r > llh0 and x_r[d] < ub[d]:
                x_r[d] = min(x_r[d] + step[d], ub[d])
                llh_r  = llh_func(x_r)

        assert np.isfinite(llh0)

        x_cur = xx.copy()
        n_steps = 0
        while True:
            xd = np.random.rand()*(x_r[d] - x_l[d]) + x_l[d]
            x_cur[d] = xd
            last_llh = llh_func(x_cur)
            if last_llh > llh0:
                xx[d] = xd
                break
            elif xd > xx[d]:
                x_r[d] = xd
            elif xd < xx[d]:
                x_l[d] = xd
            else:
                raise Exception("Slice sampler shrank too far.")
            n_steps += 1


    if not np.isfinite(last_llh):
        raise Exception("Likelihood is not finite at sampled point")

    return xx, last_llh


def slicesample(init_x, logprob, *logprob_args, **slice_sample_args):
    """generate a new sample from a probability density using slice sampling

    Parameters
    ----------
    init_x : array
    logprob : callable, `lprob = logprob(x, *logprob_args)`
        A functions which returns the log probability at a given
        location
    *logprob_args :
        additional arguments are passed to logprob

    Returns
    -------
    new_x : float
        the sampled position
    new_llh : float 
        the log likelihood at the new position
    Notes
    -----
    http://en.wikipedia.org/wiki/Slice_sampling
    """
    sigma         = slice_sample_args.get('sigma', 1.0)
    step_out      = slice_sample_args.get('step_out', True)
    max_steps_out = slice_sample_args.get('max_steps_out', 1000)
    compwise      = slice_sample_args.get('compwise', True)
    numdir        = slice_sample_args.get('numdir', 2)
    doubling_step = slice_sample_args.get('doubling_step', True)
    verbose       = slice_sample_args.get('verbose', False)
    upper_bound   = slice_sample_args.get('upper_bound', np.inf)
    lower_bound   = slice_sample_args.get('lower_bound', -np.inf)

    ## define a univariate directional slice sampler
    def direction_slice(direction, init_x):
        def dir_logprob(z):
            return logprob(direction*z + init_x, *logprob_args)

        def acceptable(z, llh_s, L, U):
            while (U-L) > 1.1*sigma:
                middle = 0.5*(L+U)
                splits = (middle > 0 and z >= middle) or (middle <= 0 and z < middle)
                if z < middle:
                    U = middle
                else:
                    L = middle
                # Probably these could be cached from the stepping out.
                if splits and llh_s >= dir_logprob(U) and llh_s >= dir_logprob(L):
                    return False
            return True

        # if we have upper/lower bounds on parameters, compute them in z space
        if compwise:
            dir_upper_bound = upper_bound[direction==1.]
            dir_lower_bound = lower_bound[direction==1.]
        else:
            dir_upper_bound = np.min(np.sign(direction)*(upper_bound - init_x)/direction)
            dir_lower_bound = np.max(np.sign(direction)*(lower_bound - init_x)/direction)
        #print np.sign(direction)*(upper_bound - init_x) / direction
        #print upper_bound
        #print np.sign(direction)
        #dir_upper_bound -= 1e-6
        #dir_lower_bound += 1e-6
        #print dir_lower_bound
        #print dir_upper_bound

        # compute initial interval bounds
        upper = sigma * npr.rand()
        lower = upper - sigma

        # sample uniformly under the probability at z = 0
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)

        # perform the stepping out or doubling procedure to compute interval I
        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            if doubling_step:
                while (dir_logprob(lower) > llh_s or dir_logprob(upper) > llh_s) and (l_steps_out + u_steps_out) < max_steps_out:
                    if npr.rand() < 0.5:
                        l_steps_out += 1
                        lower       -= (upper-lower)
                    else:
                        u_steps_out += 1
                        upper       += (upper-lower)
            else:
                while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                    l_steps_out += 1
                    lower       -= sigma
                while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                    u_steps_out += 1
                    upper       += sigma

        # uniformly sample - perform shrinkage (with accept check)
        start_upper = upper
        start_lower = lower
        steps_in = 0
        while True:
            steps_in += 1
            if steps_in % 100 == 0:
                print "shrinking, steps", steps_in
            new_z     = (upper - lower)*npr.rand() + lower
            new_llh   = dir_logprob(new_z)
            if np.isnan(new_llh):
                print new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x, *logprob_args)
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s and acceptable(new_z, llh_s, start_lower, start_upper):
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")
        if verbose:
            print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in

        return new_z*direction + init_x, new_llh

    # check to make sure we return what we hand in
    if type(init_x) == float or isinstance(init_x, np.number):
        init_x = np.array([init_x])
        scalar = True
    else:
        scalar = False

    ## expand upper and lower bound to be same dimension as samples
    if type(upper_bound) == float or isinstance(upper_bound, np.number):
        upper_bound = np.tile(upper_bound, init_x.shape)
    if type(lower_bound) == float or isinstance(lower_bound, np.number):
        lower_bound = np.tile(lower_bound, init_x.shape)

    ## make sure we're starting in a legal spot
    assert np.all(init_x < upper_bound), "init_x (%s) >= ub (%s)" % \
        (np.array_str(init_x), np.array_str(upper_bound))
    assert np.all(init_x > lower_bound), "init_x (%s) <= lb (%s)" % \
        (np.array_str(init_x), np.array_str(lower_bound))

    # do either component-wise slice sampling, or random direction
    dims = init_x.shape[0]
    if compwise:
        ordering = range(dims)
        npr.shuffle(ordering)
        new_x = init_x.copy()
        for d in ordering:
            direction    = np.zeros((dims))
            direction[d] = 1.0
            new_x, new_llh = direction_slice(direction, new_x)
    else:
        new_x = init_x
        for d in range(numdir):
            direction      = npr.randn(dims)
            direction      = direction / np.sqrt(np.sum(direction**2))
            new_x, new_llh = direction_slice(direction, new_x)

    # return sample (fix up scalar/vector outputs)
    if scalar:
        return float(new_x[0]), new_llh
    else:
        return new_x, new_llh

if __name__ == '__main__':
    npr.seed(1)

    import pylab as pl
    import pymc

    D  = 10
    fn = lambda x: -0.5*np.sum(x**2)

    iters = 1000
    samps = np.zeros((iters,D))
    #for ii in xrange(1,iters):
    #    samps[ii,:], new_llh = slicesample(samps[ii-1,:], fn, 
    #                              sigma=0.1, 
    #                              step_out=True,
    #                              doubling_step=True,
    #                              verbose=False, 
    #                              compwise=False,
    #                              numdir  =4)

    #ll = -0.5*np.sum(samps**2, axis=1)
    #scores = pymc.geweke(ll)
    #pymc.Matplot.geweke_plot(scores, 'test')
    #pymc.raftery_lewis(ll, q=0.025, r=0.01)
    #pymc.Matplot.autocorrelation(ll, 'test')

    #import seaborn as sns
    #sns.jointplot(samps[:,0], samps[:,1])
    #plt.show()


    # bounded test
    def fn(x): 
        if np.any(x <= 0.):
            return -np.inf
        return -.5*np.sum(x**2)
    samps = np.zeros((iters,D))
    samps[0,:] = np.random.rand(D)
    for ii in xrange(1,iters):
        samps[ii,:], new_llh = slicesample(samps[ii-1,:], fn, 
                                  sigma=0.1, 
                                  step_out=True,
                                  doubling_step=True,
                                  verbose=True, 
                                  compwise=False,
                                  lower_bound = 0.,
                                  numdir  = 4)

    import seaborn as sns
    sns.jointplot(samps[:,0], samps[:,1])
    plt.show()




    

