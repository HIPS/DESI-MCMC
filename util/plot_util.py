import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('..'), os.path.pardir)))
from celeste import gen_model_image

def plot_comparison(band, file_ra_dec, model_image):   
    ## 0. Create an image
    fits_file_template = "../data/blobs/stamp-%s-%s.fits"%("%s", file_ra_dec)
    fits_img = FitsImage(band, fits_file_template) #="../data/blobs/stamp-%s-305.1114--12.9577.fits")

    ## 1. guess the location of the source find brightest spot in pixel space 
    #s_pix  = np.array(np.where(fits_img.nelec.max()==fits_img.nelec)).squeeze()
    #print "Maximally brightest pixel", s_pix
    #s_equa = fits_img.pixel2equa(s_pix)

    cat_file = "../data/blobs/cat-%s.fits"%file_ra_dec
    srcs = get_sources_from_catalog(cat_file = cat_file) #'../data/blobs/cat-305.1114--12.9577.fits')
    print "%d sources in catalog"%len(srcs)
    #srcs[1].u = np.array([130.17437611,  52.75034404])
    #srcs[1].b[fits_img.band] = 107.62

    ## 2. Plot model image
    fig, axarr  = plt.subplots(1, 3)
    axarr[0].imshow(fits_img.nelec.T, interpolation='none', origin='lower')
    axarr[0].set_title('Observed Image, band=%s'%fits_img.band)
    axarr[1].imshow(model_image.T, interpolation='none', origin='lower')
    axarr[1].set_title('Model Image, band=%s'%fits_img.band)
    im = axarr[2].imshow(model_image.T - fits_img.nelec.T, interpolation='none', origin='lower')
    axarr[2].set_title('Model - Observed')
    print "absolute max distance between model and observed %d"%( np.abs(fits_img.nelec - model_image).max() )
    print "max brightness observed: %d"%fits_img.nelec.max()
    print "max brightness model   : %d"%model_image.max()
    plt.colorbar(im)
    return fig, axarr


def compare_to_model(srcs, img, fig=None, axarr=None): 
    """ plots true image, model image, and difference (much like above) 
        Input: 
            srcs: python list of PointSrcParams
            img : FitsImage object

        Output:
            fig, axarr
    """
    if fig is None or axarr is None:
        fig, axarr = plt.subplots(1, 3)

    # generate model image
    model_img = gen_model_image(srcs, img)
    vmin = min(img.nelec.min(), model_img.min())
    vmax = max(img.nelec.max(), model_img.max())

    im1 = axarr[0].imshow(np.log(img.nelec), interpolation='none', origin='lower', vmin=np.log(vmin), vmax=np.log(vmax))
    axarr[0].set_title('log data ($\log(x_{n,m})$')
    im2 = axarr[1].imshow(np.log(model_img), interpolation='none', origin='lower', vmin=np.log(vmin), vmax=np.log(vmax))
    axarr[1].set_title('log model ($\log(F_{n,m})$)')
    divider2 = make_axes_locatable(axarr[1])
    cax2 = divider2.append_axes('right', size='10%', pad=.05)
    cbar2 = fig.colorbar(im2, cax=cax2)

    im3 = axarr[2].imshow(img.nelec-model_img, interpolation='none', origin='lower')
    axarr[2].set_title('Difference: Data - Model')
    divider3 = make_axes_locatable(axarr[2])
    cax3 = divider3.append_axes('right', size='10%', pad=.05)
    cbar3 = fig.colorbar(im3, cax=cax3)
    return fig, axarr

