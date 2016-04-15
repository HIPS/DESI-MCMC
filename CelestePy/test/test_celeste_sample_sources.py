import CelestePy.celeste_sample_sources as css
import numpy as np

def test_sample_binomial():

    N = 914
    p = 6.29379e-16
    print css.sample_binomial(N, p, np.random.RandomState(0))

