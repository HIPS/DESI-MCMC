""" 
Unit tests for convolution functions 
"""
import sys
sys.path.append("..")
import numpy as np
from scipy.signal import convolve as sconvolve
import matplotlib.pyplot as plt
import conv_net as cv

def test_im2col_convolve():
    """ compares my im2col based dot product convolve with scipy convolve """
    skip = 1
    block_size = (11, 11)
    img   = np.random.randn(227, 227, 3)
    filt  = np.dstack([cv.gauss_filt_2D(shape=block_size,sigma=2) for k in range(3)])

    # im2col the image and filter
    img_cols = cv.im2col(img, block_size=block_size, skip=skip)
    out      = cv.convolve_im2col(img_cols, filt, block_size, skip, img.shape)

    # check against scipy convolve
    outc = np.dstack([ sconvolve(img[:,:,k], filt[:,:,k], mode='valid') for k in range(3)])
    outc = np.sum(outc, axis=2)
    assert np.allclose(out, outc), "im2col skip 1 failed!"

def test_im2col_convolve_skip():
    """ compares my im2col based dot product convolve with scipy convolve """
    skip = 5
    block_size = (11, 11)
    img   = np.random.randn(227, 227, 3)
    filt  = np.dstack([cv.gauss_filt_2D(shape=block_size,sigma=2) for k in range(3)])

    # im2col the image and filter
    img_cols = cv.im2col(img, block_size=block_size, skip=skip)
    out      = cv.convolve_im2col(img_cols, filt, block_size, skip, img.shape)

    # check against scipy convolve
    outc = np.dstack([ sconvolve(img[:,:,k], filt[:,:,k], mode='valid') for k in range(3)])
    outc = np.sum(outc, axis=2)
    outc = outc[::skip, ::skip]
    assert np.allclose(out, outc), "im2col skip 1 failed!"


