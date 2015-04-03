import numpy as np
def slicesample(xx, llh_func, last_llh=None, step=1, step_out=True, x_l=None, x_r=None, lb=-np.Inf, ub=np.Inf):
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

    assert np.all(xx < ub)
    assert np.all(xx > lb)

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

