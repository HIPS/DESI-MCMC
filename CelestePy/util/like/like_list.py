import numpy as np

"""
List of simple log pdfs
"""
def fast_gamma_lnpdf(x, a0, b0): 
    """ Unnormalized gamma log pdf.  a0 = shape, b0 = rate """
    logprobs = (a0 - 1.) * np.log(x) - b0*x
    if not isinstance(x, (list, tuple, np.ndarray)):
        if x <= 0.: 
            return -np.inf
        return logprobs

    # for
    logprobs[x <= 0.] = -np.inf
    return logprobs 

def fast_inv_gamma_lnpdf(x, a0, b0): 
    """ unnormalized inverse gamma log pdf.  a0 = shape, b0 = inv scale"""
    logprobs = (-a0 - 1.) * np.log(x) - b0 / x
    if not isinstance(x, (list, tuple, np.ndarray)):
        if x <= 0.:
            return -np.inf
        return logprobs

    logprobs[x <= 0.] = -np.inf
    return logprobs

def fast_normal_lnpdf(x, mu, sig2):
    return .5 * (1./sig2) * (x - mu)*(x - mu)



if __name__=="__main__":
    xgrid = np.linspace(0, 10, 100)
    ggrid = fast_gamma_lnpdf(xgrid, 1., 1.)
    igrid = fast_inv_gamma_lnpdf(xgrid, 1., 1.)
    plt.plot(xgrid, np.exp(ggrid))
    plt.plot(xgrid, np.exp(igrid))
    plt.show()
