import tractor.sdss as sdss
from tractor.basics import PointSource
from tractor.galaxy import ExpGalaxy, DevGalaxy, CompositeGalaxy
from CelestePy import gen_point_source_psf_image, \
                      gen_galaxy_psf_image, FitsImage, SrcParams
import numpy as np
import matplotlib.pyplot as plt

def tsrc_to_src_params(tsrc):
    pos = tsrc.getPosition()
    u = [p for p in pos]
    fluxes = tsrc.getBrightnesses()[0]
    fluxes = [flux for flux in fluxes]

    if type(tsrc) == PointSource:
        return SrcParams(u, a=0, fluxes=fluxes)
    else:
        if type(tsrc) == ExpGalaxy:
            shape = tsrc.getShape()
            theta = 0.
        elif type(tsrc) == DevGalaxy:
            shape = tsrc.getShape()
            theta = 1.
        elif type(tsrc) == CompositeGalaxy:
            shape = tsrc.shapeExp
            theta = 0.5
        else:
            pass

        return SrcParams(u,
                         a=1,
                         v=u,
                         theta=theta,
                         phi=shape[2],
                         sigma=shape[0],
                         rho=shape[1],
                         fluxes=fluxes)

BANDS = ['u', 'g', 'r', 'i', 'z']
def main(run, camcol, field):
    # read in sources, images
    srcs = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgs = {}
    for band in BANDS:
        print "reading in band %s" % band
        imgs[band] = sdss.get_tractor_image_dr9(run, camcol, field, band)

    # convert to FitsImage's
    imgfits = {}
    for band in BANDS:
        print "converting images %s" % band
        imgfits[band] = FitsImage(band, timg=imgs[band])

    # get images
    modelims = {}
    for i,src in enumerate(srcs[0:50]):
        print "Source %d" % i
        # convert to Celeste sources
        src_params = tsrc_to_src_params(src)

        for j,band in enumerate(BANDS):
            if src_params.a == 0:
                f_s = gen_point_source_psf_image(src_params.u, imgfits[band])
            elif src_params.a == 1:
                f_s = gen_galaxy_psf_image(src_params, imgfits[band]);

            if band in modelims:
                modelims[band] += f_s * src_params.fluxes[j]
            else:
                modelims[band] = f_s * src_params.fluxes[j]

    return modelims

if __name__ == '__main__':
    run = 125
    camcol = 1
    field = 17
    modelims = main(run, camcol, field)

