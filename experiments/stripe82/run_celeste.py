import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
plt.ion()
from CelestePy.util.data import make_fits_images, tractor_src_to_celestepy_src
import numpy as np
import pandas as pd
import pyprind

#stripe 82 files
STRIPE_82_DATA_DIR = "../../data/stripe_82_dataset/"
import sys, os
sys.path.append(STRIPE_82_DATA_DIR)
from load_stripe82_square import df_from_fits, create_matched_dataset


def examine_pixel_error(model):
    # stack equatorial locs for each source
    U = np.array([s.params.u for s in model.srcs])
    stypes = np.array([s.object_type for s in model.srcs])

    # compute first order approximation,
    rimg = model.field_list[0].img_dict['r']
    approx_pixel = np.array([rimg.equa2pixel(s.params.u) for s in model.srcs])

    # compute wcs pixel center
    from astropy import wcs
    wcs_r = wcs.WCS(rimg.header)
    wcs_pixel   = wcs_r.wcs_world2pix(U, 0)

    # SANITY CHECK - make sure the initialized source locations fall within the image
    max_y, max_x = rimg.nelec.shape
    num_in_box = np.sum( (wcs_pixel[:,0] > 0) & (wcs_pixel[:,0] < max_x) &
                         (wcs_pixel[:,1] > 0) & (wcs_pixel[:,1] < max_y))
    print "  ... %d sources found in image bounds (of %d)"%(num_in_box, wcs_pixel.shape[0])

    # look at close to center and not close to center images
    img_center = np.array([max_x/2., max_y/2.])
    dist_to_center = np.sqrt(np.sum((wcs_pixel - img_center)**2, axis=1))
    pixel_error    = np.sqrt(np.sum( (approx_pixel - wcs_pixel)**2, axis=1))

    # look at pixel error as a function of dist from center
    plt.scatter(dist_to_center, pixel_error)
    plt.ylim((pixel_error.min(), pixel_error.max()))
    plt.xlim((dist_to_center.min(), dist_to_center.max()))
    plt.xlabel("Distance to field center (pixels)")
    plt.ylabel("First Order Approx Error (pixels)")
    plt.title("WCS Pixel Error By Field Location")
    plt.savefig("figs/pixel_error.pdf", bbox_inches='tight')
    plt.close("all")

    # plot two model images the model
    star_idxs = np.where(stypes == 'star')[0]
    star_dists = dist_to_center[star_idxs]
    close_i  = star_idxs[np.argsort(star_dists)[0]]
    far_i    = star_idxs[np.argsort(star_dists)[-1]]
    fig, axarr = plt.subplots(2, 3, figsize=(12, 12))
    for ii, i in enumerate([close_i, far_i]):
        model.srcs[i].plot(rimg, ax      = axarr[ii,0],
                                 data_ax = axarr[ii,1],
                                 diff_ax = axarr[ii,2])
        axarr[ii,0].set_title("Source model (dist to center = %2.2f, type=%s)" % \
                        (dist_to_center[i], model.srcs[i].object_type))
        axarr[ii,1].set_title("Source data")
        axarr[ii,2].set_title("(scaled) diff")
    plt.savefig("figs/initial_model_comparison.pdf", bbox_inches='tight')


def examine_source_location_likelihood(src):
    # visualize likelihood surface around center (10 pixels each way)
    pix_u = imgfits['r'].equa2pixel(src.paramsrc.u)
    xmin, ymin = imgfits['r'].pixel2equa(pix_u-20)
    xmax, ymax = imgfits['r'].pixel2equa(pix_u+20)

    xgrid = np.linspace(xmin, xmax, 50)
    ygrid = np.linspace(ymin, ymax, 50)
    xx, yy = np.meshgrid(xgrid, ygrid)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    ll  = np.array([src.location_likelihood(pt) for pt in pyprind.prog_bar(pts)])
    fig = plt.figure(figsize=(8,8))
    plt.contourf(xx, yy, ll.reshape(xx.shape))
    plt.colorbar()
    plt.title("Location Likelihood for source")
    plt.savefig('figs/location_likelihood.pdf', bbox_inches='tight')


def examine_brightest_sources(model):
    # grab brightest star, brightest gal
    fig, axarr = plt.subplots(2, 3, figsize=(12, 12))
    rimg = imgfits['r']
    ts = ['star', 'galaxy']
    for i, t in enumerate(ts):
        src = model.get_brightest(object_type=t)[0]
        src.plot(rimg, ax=axarr[i,0], data_ax=axarr[i,1], diff_ax=axarr[i,2])
        axarr[i,0].set_title("Source model (type=%s)"%model.srcs[i].object_type)
        axarr[i,1].set_title("Source data")
        axarr[i,2].set_title("(scaled) diff")
    plt.savefig("figs/brightest_sources.pdf", bbox_inches='tight')


def examine_initialization(s, imgfits, model, run, camcol, field, id=None, band='r'):
    if id is None:
        id = s.id
    fimg = imgfits[band]

    # plot initial state of s
    fig, axarr = plt.subplots(4, 3, figsize=(10,10))
    s.plot(fimg, *axarr[0])

    # plot tractor initialization 
    from CelestePy.util.misc import plot_util
    import tractor_render as tr
    _, ylim, xlim = s.compute_model_patch(fimg)
    def render_tractor_patch(tractor_patch, tim, xlim, ylim, axarr):
        # rescale tractor patch and img data by fimg details
        def img_nano2count(patch, fimg):
            return np.round(patch / fimg.calib * fimg.kappa)

        tractor_patch = img_nano2count(tractor_patch, fimg)
        tim_count     = img_nano2count(tim.data, fimg)

        cim = axarr[0].imshow(tractor_patch, extent=xlim+ylim)
        plot_util.add_colorbar_to_axis(axarr[0], cim)
        cim = axarr[1].imshow(tim_count, extent=xlim+ylim)
        plot_util.add_colorbar_to_axis(axarr[1], cim)
        cim = axarr[2].imshow(tim_count-tractor_patch, extent=xlim+ylim)
        plot_util.add_colorbar_to_axis(axarr[2], cim)
        axarr[2].set_title("diff, mse = %2.3f"%( np.mean((tim_count-tractor_patch)**2)))
        axarr[0].set_title("tractor")

    # render with tractor catalog
    tractor_patch, tim = tr.tractor_render_patch(run, camcol, field, radec=s.params.u,
                                roi=xlim+ylim, celeste_src=None, bandname=band)
    render_tractor_patch(tractor_patch, tim, xlim, ylim, axarr[1])
    axarr[1,0].set_title("Tractor (tractor catalog)")
    tractor_patch, tim = tr.tractor_render_patch(run, camcol, field, radec=s.params.u,
                                roi=xlim+ylim, celeste_src=s, bandname=band)
    render_tractor_patch(tractor_patch, tim, xlim, ylim, axarr[2])
    axarr[2,0].set_title("Tractor (photo params)")

    # resample photon images and plot
    for _ in xrange(0):
        model.field_list[0].resample_photons([s])
        s.resample_star()
        print "marg like: ", s.log_likelihood_isolated()
    s.plot(fimg, *axarr[3])

    # label y axes for each plot
    axarr[0,0].set_ylabel('photo init (celeste render)')
    axarr[1,0].set_ylabel('tractor catalog (tractor render)')
    axarr[2,0].set_ylabel('photo params (tractor render)')
    axarr[3,0].set_ylabel('celeste em (celeste render)')

    # show plot
    #fig.suptitle("Photo (top) vs Celeste (bottom) initializations")
    #fig.tight_layout()
    plt.savefig("figs/source_inits/%s_init.pdf"%id, bbox_inches='tight')


def report_star_error(bsrcs, bidx, primary_field_df, coadd_field_df):
    """take sources we sampled, report how good flux distributions were """
    from CelestePy.util.data import nanomaggies2mags

    #### compare flux mags ####
    mag_ests = np.array([ nanomaggies2mags(s.flux_samples).mean(0)
                         for s in bsrcs])
    bands = ['u', 'g', 'r', 'i', 'z']
    primary_mags = primary_field_df[['psfMag_%s'%b for b in bands]].values[bidx,:]
    coadd_mags   = coadd_field_df[['psfMag_%s'%b for b in bands]].values[bidx,:]
    celeste_flux_resid = coadd_mags - mag_ests
    primary_flux_resid = coadd_mags - primary_mags
    flux_rmse = np.sqrt(np.row_stack([ np.mean(celeste_flux_resid**2, axis=0),
                                       np.mean(primary_flux_resid**2, axis=0)]))

    #### compare locations ####
    loc_ests     = np.array([s.location_samples.mean(0) for s in bsrcs])
    primary_locs = primary_field_df[['ra', 'dec']].values[bidx,:]
    coadd_locs   = coadd_field_df[['ra', 'dec']].values[bidx,:]
    celeste_loc_resid = coadd_locs - loc_ests
    primary_loc_resid = coadd_locs - primary_locs
    loc_rmse = np.sqrt(np.row_stack([np.mean(celeste_loc_resid**2, axis=0),
                                     np.mean(primary_loc_resid**2, axis=0)]))

    # big rmse mat and percent improvement 
    rmse_mat = np.column_stack([flux_rmse, loc_rmse])
    percent_improvement = (rmse_mat[1] - rmse_mat[0]) / rmse_mat[1]

    cols = bands + ['ra', 'dec']
    error_df = pd.DataFrame(np.row_stack([rmse_mat, percent_improvement]), columns=cols)
    error_df.index = ['celeste', 'primary', '% improvement']
    print "RMSE table", error_df
    return error_df


if __name__ == '__main__':

    ###################################################
    # load in each catalog file as a pandas dataframe #
    ###################################################
    test_primary_fn = os.path.join(STRIPE_82_DATA_DIR, "square_4263_4.fit")
    test_coadd_fn   = os.path.join(STRIPE_82_DATA_DIR, "square_106_4.fit")
    primary_df      = df_from_fits(test_primary_fn)
    coadd_df        = df_from_fits(test_coadd_fn)

    # create a matched dataset - coadd source (ground truth) to 
    # primary sources (baseline)
    primary_matched, coadd_matched, dists = \
    	create_matched_dataset(primary_df, coadd_df)

    #################################################
    # look at the breakdown by field
    #################################################
    print "Stripe 82 dataset statistics:"
    field_info = primary_matched[['run', 'camcol', 'field']].drop_duplicates()
    for field in np.sort(field_info.field):
        primary_field_df = primary_matched[primary_matched.field == field]
        num_stars        = np.sum(primary_field_df.type==6)
        num_gals         = np.sum(primary_field_df.type==3)
        print "    field %d: %d stars, %d galaxies (%d total)" % \
                (field, num_stars, num_gals, primary_field_df.shape[0])

    ########################################################
    # subselect stripe field 367 - get existing sources
    ########################################################
    run, camcol, field = 4263, 4, 367
    idx = np.where(primary_matched.field == field)[0]
    #primary_field_df = primary_matched[primary_matched.field == field]
    primary_field_df = primary_matched.iloc[idx]
    coadd_field_df   = coadd_matched.iloc[idx]

    from tractor import sdss as st
    tsrcs = st.get_tractor_sources_dr9(run, camcol, field)

    # grab fits images 
    imgfits = make_fits_images(run, camcol, field)

    #############################################
    # initialize celeste model
    #############################################
    import CelestePy.model_sources as models
    reload(models)
    model = models.CelesteGMMPrior()

    # for each run/camcol/field, add a data
    model.add_field(img_dict = imgfits)

    # initialize sources from photo obj catalog
    model.initialize_sources(photoobj_df = coadd_field_df)
    #model.initialize_sources(init_src_params = [tractor_src_to_celestepy_src(s) for s in tsrcs])

    # get brightest-ish sources
    ssrcs, sidx = model.get_brightest(object_type='star', num_srcs=40, return_idx=True)
    gsrcs, gidx = model.get_brightest(object_type='galaxy', num_srcs=40, return_idx=True)
    bsrcs = ssrcs[35:] + gsrcs[17:20]
    bidx  = np.concatenate([sidx[35:], gidx[17:20]])

    # breadcrumbs - make sure we can examine which source corresponds to
    # which catalog entry
    blocs = np.array([s.params.u for s in bsrcs])
    plocs = primary_field_df[['ra', 'dec']].values[bidx,:]
    assert np.allclose(blocs, plocs), "not the same location! noooo"

    ######################################
    # gibbs step on a handful of sources #
    ######################################
    print "======= running celeste sampler ========"
    # do some resampling, each source keeps each sample
    Nsamps = 1
    for i in pyprind.prog_bar(xrange(Nsamps)):

        # resample photon images
        model.field_list[0].resample_photons(bsrcs)

        # resample one star and one galaxy
        #fig, axarr = plt.subplots(2, 3)
        #star = bsrcs[0]
        #gal  = bsrcs[-1]
        #gal.plot(imgfits['r'], *axarr[0])
        #gal.resample()
        #gal.plot(imgfits['r'], *axarr[1])
        #t_us = np.array([ np.array(ts.getPosition()) for ts in tsrcs])
        #dists = np.sum((t_us - gal.params.u)**2, axis=1)
        #ts = tsrcs[np.argmin(dists)]

        # resample source params
        for s in pyprind.prog_bar(bsrcs):
            s.resample()
            s.store_sample()
            s.store_loglike()

        # global/local update
        #for s in bsrcs:
        #    s.sample_type()

        # global updates
        #model.sample_birth()
        #model.sample_death()

    ##########################
    # DEBUG
    # look at pixel error and a few plots based on distance and source fluxes
    if False:
        examine_pixel_error(model)
        examine_brightest_sources(model)

    sys.exit()

    if False:
        for bright_i, s in enumerate(bsrcs):
            examine_initialization(s, id="bright_%d"%bright_i, imgfits=imgfits, model=model, run=run, camcol=camcol, field=field)
            plt.close("all")

        rmod_img = model.render_model_image(imgfits['r'])

        # visualize galaxy
        # get galaxies with rho close to 1
        gals = [s for s in model.srcs if s.is_galaxy()]
        ecc_order = np.argsort([s.params.rho for s in gals])
        top_ecc   = [gals[i] for i in ecc_order[:300]]
        rs = [s.params.flux_dict['r'] for s in top_ecc]
        rs_i = np.argsort(rs)[::-1]
        examine_initialization(top_ecc[rs_i[2]], imgfits=imgfits, model=model, run=run, camcol=camcol, field=field)

        gsrcs, bidx = model.get_brightest(object_type='galaxy', num_srcs=20, return_idx=True)
        for bright_i, s in enumerate(gsrcs):
            examine_initialization(s, id="bright_%d"%bright_i, imgfits=imgfits, model=model, run=run, camcol=camcol, field=field)
            plt.close("all")

    ###END DEBUG #######################


    ######################################
    # Model Evaluation                   #
    ######################################
    fig, axarr = plt.subplots(1,3)
    bsrcs[-1].plot(imgfits['r'], *axarr)
    plt.show()

    plt.plot(bsrcs[-1].loglike_samples)
    plt.show()

    plt.close("all")

    report_star_error(bsrcs, bidx, primary_field_df, coadd_field_df)

    def plot_source_histogram(src):
        pass


    ###############################################
    # Debug visualize Source Location Likelihoods #
    ###############################################

    # look at single source conditional likelihoods
    examine_source_location_likelihood(model.srcs[-5])
    plt.show()

    sys.exit()

    # debug
    reload(models)
    #s = model.srcs[0]
    src_i = -7
    s = models.Source(model.srcs[src_i].params)
    print s.object_type
    print s.params.u
    s.sample_image_list = model.srcs[src_i].sample_image_list

    from CelestePy.util.infer.slicesample import slicesample
    u = s.params.u.copy()
    us = np.zeros((100, 2))
    us[0,:] = s.params.u
    for i in pyprind.prog_bar(xrange(1,us.shape[0])):
        us[i,:] = s.resample_location(u=us[i-1,:])

    grid = sns.jointplot(us[:,0], us[:,1])
    grid.ax_joint.set_xlim(us[:,0].min(), us[:,0].max())
    grid.ax_joint.set_ylim(us[:,1].min(), us[:,1].max())
    plt.show()

    from scipy.optimize import minimize
    res = minimize(lambda u: -1.*s.location_likelihood(u), x0=s.params.u.copy(), method='Nelder-Mead')
    print "Coadd loc: ", coadd_field_df[['ra', 'dec']].values[src_i]
    print "Primary Loc:", primary_field_df[['ra', 'dec']].values[src_i]
    print "Celeste samp", us.mean(0)
    print "Celeste opt", res.x
    #model.srcs[0].resample()



