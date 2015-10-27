""" 
Unit tests for convolution functions 
"""
import sys
sys.path.append("..")
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.util import quick_grad_check
from scipy.signal import convolve as sconvolve
from autograd.scipy.signal import convolve as auto_convolve
import conv_net as cv
import fast_conv as fc

def test_fast_conv():
    """ compares my fast_conv to scipy convolve """
    skip = 1
    block_size = (11, 11)
    depth = 5
    img   = np.random.randn(51, 51, depth)
    filt  = np.dstack([cv.gauss_filt_2D(shape=block_size,sigma=2) for k in range(depth)])

    # im2col the image and filter
    out = fc.convolve(filt, img)

    # check against scipy convolve
    outc = np.dstack([ auto_convolve(img[:,:,k], filt[:,:,k], mode='valid') for k in range(3)])
    outc = np.sum(outc, axis=2)
    assert np.allclose(out, outc), "fast_conv (cythonized) failed!"

def test_fast_conv_grad():
    skip = 1
    block_size = (11, 11)
    depth = 1
    img   = np.random.randn(51, 51, depth)
    filt  = np.dstack([cv.gauss_filt_2D(shape=block_size,sigma=2) for k in range(depth)])
    filt = cv.gauss_filt_2D(shape=block_size, sigma=2)
    def loss_fun(filt):
        out = fc.convolve(filt, img)
        return np.sum(np.sin(out) + out**2)
    loss_fun(filt)
    loss_grad = grad(loss_fun)

    def loss_fun_slow(filt):
        out = auto_convolve(img.squeeze(), filt, mode='valid') 
        return np.sum(np.sin(out) + out**2)
    loss_fun_slow(filt)
    loss_grad_slow = grad(loss_fun_slow)

    # compare gradient timing
    loss_grad_slow(filt)
    loss_grad(filt)

    ## check numerical gradients
    num_grad = np.zeros(filt.shape)
    for i in xrange(filt.shape[0]):
        for j in xrange(filt.shape[1]):
            de = np.zeros(filt.shape)
            de[i, j] = 1e-4
            num_grad[i,j] = (loss_fun(filt + de) - loss_fun(filt - de)) / (2*de[i,j])

    assert np.allclose(loss_grad(filt), num_grad), "convolution gradient failed!"


#def test_im2col_convolv_grad():
#    skip = 1
#    block_size = (11, 11)
#    img   = np.random.randn(227, 227, 3)
#    filt  = np.dstack([cv.gauss_filt_2D(shape=block_size,sigma=2) for k in range(3)])
#
#    # im2col the image and filter
#    img_cols = cv.im2col(img, block_size=block_size, skip=skip)
#    out      = cv.convolve_im2col(img_cols, filt, block_size, skip, img.shape)
#
#    # gradient of convolution with respect to a scalar function
#    def loss_fun(filt):
#        return np.sum(cv.convolve_im2col(img_cols, filt, block_size, skip, img.shape))
#
#    loss_grad = grad(loss_fun)
#    loss_grad(filt)

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


