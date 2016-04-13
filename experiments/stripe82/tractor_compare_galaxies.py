"""
Compares Celeste and Tractor's galaxy rendering
"""
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
plt.ion()
from CelestePy.util.data import make_fits_images, tractor_src_to_celestepy_src
import numpy as np
import pandas as pd
import pyprind


if __name__ == '__main__':

    ########################################################
    # subselect stripe field 367 - get existing sources
    ########################################################
    run, camcol, field = 4263, 4, 367
    from tractor import sdss as st
    tsrcs = st.get_tractor_sources_dr9(run, camcol, field)

     # grab fits images 
    imgfits = make_fits_images(run, camcol, field)

    #######################################################################
    # find a bright Exp galaxy, inspect celeste params and tractor params #
    #######################################################################
    import tractor.galaxy as tg
    import tractor.pointsource as ps
    def is_gal(s):
        return  (type(s) == tg.DevGalaxy) or \
                (type(s) == tg.ExpGalaxy) or \
                (type(s) == tg.CompositeGalaxy)

    def shape(s):
        if type(s) == tg.DevGalaxy or type(s) == tg.ExpGalaxy:
            return s.getShape()
        elif type(s) == tg.CompositeGalaxy:
            return s.shapeDev

    gals = [s for s in tsrcs if (type(s)==tg.ExpGalaxy) and
                                s.getBrightness()[2]<20 and
                                shape(s).ab < .4 ]
    stars = [s for s in tsrcs if (type(s) == ps.PointSource) and
                                  s.getBrightness()[2]<20]
    print len(gals)

    ##########################################################
    # initialize celeste model with small Exp gal collection
    ##########################################################
    import CelestePy.model_sources as models
    reload(models)
    model = models.CelesteGMMPrior()
    model.add_field(img_dict = imgfits)
    model.initialize_sources(
        init_src_params = [tractor_src_to_celestepy_src(s) for s in gals] + 
                          [tractor_src_to_celestepy_src(s) for s in stars])

    tsrcs   = gals + stars
    gal_idx = len(gals) + 1
    s       = model.srcs[gal_idx]
    s.params.theta = 1.
    ts      = tsrcs[gal_idx]

    # plot celeste/tractor comparison
    fig, axarr = plt.subplots(1, 3)
    fimg = imgfits['r']

    ###########################
    # plot celeste rendering  #
    ###########################
    simg, ylim, xlim = s.compute_scatter_on_pixels(fimg)
    print "\n========== celeste render ===============\n"
    s.plot(fimg, ax=axarr[0], unit_flux=True)
    axarr[0].set_title("Celeste Render")

    ##########################
    # plot tractor rendering #
    #########################
    print "\n========== tractor render ===============\n"
    from tractor import sdss as st
    from tractor import Tractor
    from astrometry.sdss import DR9
    # get images
    sdss     = DR9(basedir='sdss_data')
    tim,tinf = st.get_tractor_image_dr9(run, camcol, field, fimg.band,
        curl=True, roi=(xlim[0], xlim[1]-1, ylim[0], ylim[1]-1), sdss=sdss, psf='kl-gm')

    #patch = ts.getModelPatch(tim)
    px, py = fimg.equa2pixel(s.params.u) - np.array([xlim[0], ylim[0]])
    patch = ts.getUnitFluxModelPatch(tim) #, px, py)
    mod   = patch.getImage()
    # get corresponding sources
    from CelestePy.util.misc import plot_util
    #tractor = Tractor(images = [tim], catalog=[ts])
    #mod     = tractor.getModelImage(tim, srcs=[ts])
    cim = axarr[1].imshow(mod, extent=xlim+ylim)
    plot_util.add_colorbar_to_axis(axarr[1], cim)
    axarr[1].set_title("Tractor Render")

    cim = axarr[2].imshow(mod - simg)
    plot_util.add_colorbar_to_axis(axarr[2], cim)
    axarr[2].set_title("Tractor Model - Celeste Model")

    print "Sanity check parameters"
    print "\nCeleste Source: \n", s
    print "\nTractor source: \n", ts


    ####### step by step break down of galaxy renderer
    cpx, cpy = fimg.equa2pixel(s.params.u)
    tpx, tpy = tim.getWcs().positionToPixel(ts.getPosition())

    import CelestePy.celeste_galaxy_conditionals as gal_funs
    cG = gal_funs.gen_galaxy_ra_dec_basis(s.params.sigma, s.params.rho, s.params.phi)
    tG = ts.getShape().getRaDecBasis()

    # compare cd matrix
    cCD = fimg.cd_at_pixel(cpx, cpy)
    tCD = tim.getWcs().cdAtPixel(cpx - xlim[0], cpy)


    ###########################
    # plot full flux images   #
    ###########################
    # plot celeste/tractor comparison
    fig, axarr = plt.subplots(1, 3)
    fimg = imgfits['r']
    simg, ylim, xlim = s.compute_scatter_on_pixels(fimg)
    print "\n========== celeste render ===============\n"
    fig = plt.figure()
    plt.imshow(s.params.flux_dict['r']*simg); plt.colorbar(); plt.title("celeste")
    #model.srcs[1].plot(fimg, ax=axarr[0], unit_flux=True)
    #axarr[0].set_title("Celeste Render")

    # plot tractor rendering
    print "\n========== tractor render ===============\n"
    # get images
    sdss     = DR9(basedir='sdss_data')
    tim,tinf = st.get_tractor_image_dr9(run, camcol, field, fimg.band,
        curl=True, roi=(xlim[0], xlim[1]-1, ylim[0], ylim[1]-1), sdss=sdss, psf='kl-gm')

    #patch = ts.getModelPatch(tim)
    counts = tim.getPhotoCal().brightnessToCounts(ts.brightness)
    px, py = fimg.equa2pixel(s.params.u) - np.array([xlim[0], ylim[0]])
    patch = ts.getModelPatch(tim) #, px, py) #UnitFluxModelPatch(tim, px, py)
    mod   = patch.getImage()
    # get corresponding sources
    from CelestePy.util.misc import plot_util
    fig = plt.figure()
    plt.imshow(mod); plt.colorbar(); plt.title("tractor")
    ##cim = axarr[1].imshow(mod, extent=xlim+ylim)
    #plot_util.add_colorbar_to_axis(axarr[1], cim)
    #axarr[1].set_title("Tractor Render")

    fig = plt.figure()
    plt.imshow(mod-s.params.flux_dict['r']*simg); plt.colorbar(); plt.title("diff")
    #cim2 = axarr[2].imshow(mod - simg)
    plot_util.add_colorbar_to_axis(axarr[2], cim2)
    axarr[2].set_title("Tractor Model - Celeste Model")
    plt.show()
    print "Sanity check parameters"
    print "\nCeleste Source: \n", s
    print "\nTractor source: \n", ts



