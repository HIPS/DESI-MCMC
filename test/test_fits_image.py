import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('..'), os.path.pardir)))
import numpy as np
from fits_image import FitsImage

def print_test(name):
    print "=========================================================="
    print name

def test_coordinate_transformations():
    print_test("coordinate transformations")

    # load file
    fits_file_template = '../data/stamps/stamp-%s-253.1147-11.6072.fits'
    img = FitsImage('z', fits_file_template)

    # for testing coordinate transformation
    from astropy.wcs import WCS
    wcs = WCS(img.band_file)

    ## test pixel 2 equa on many pixels about the reference pixel
    fail_cnt = 0
    for i in range(-10, 60, 10): 
        for j in range(-10, 60, 20):
            s_pixel = [i, j]
            r, d, = wcs.wcs_pix2world(s_pixel[0], s_pixel[1], 1)
            u_equa = img.pixel2equa(s_pixel)
            if not np.allclose([r, d], u_equa):
                fail_cnt += 1
                print "  pixel2equa FAILS on ", s_pixel

            ## test equa 2 pixel
            u_equa  = np.array([r, d])
            x, y    = wcs.wcs_world2pix(u_equa[0], u_equa[1], 1)
            u_pixel = img.equa2pixel(u_equa)
            if not np.allclose([x, y], u_pixel, atol=1e-3): # and np.allclose(s_pixel, u_pixel)): 
                fail_cnt += 1
                print "  equa2pixel FAILS on ", s_pixel
                print "      %2.5f  =>  %2.5f"%(x, u_pixel[0])
                print "      %2.5f  =>  %2.5f"%(y, u_pixel[1])
    if fail_cnt == 0:
        print "  pixel2equa and equa2pixel pass on all test pixels"


if __name__=="__main__":
    test_coordinate_transformations()


