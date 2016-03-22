import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
plt.ion()
from CelestePy.util.data import make_fits_images
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


def examine_initialization(s, imgfits, model, id=None):
    if id is None:
        id = s.id

    # plot initial state of s
    fig, axarr = plt.subplots(2, 3, figsize=(10,10))
    s.plot(imgfits['i'], *axarr[0])

    # resample photon images
    for _ in xrange(5):
        model.field_list[0].resample_photons([s])
        s.resample_star()
        print "marg like: ", s.log_likelihood_isolated()

    # plot post samples and save
    s.plot(imgfits['i'], *axarr[1])
    fig.suptitle("Photo (top) vs Celeste (bottom) initializations")
    fig.tight_layout()
    plt.savefig("figs/source_inits/%s_init.pdf"%s.id, bbox_inches='tight')


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

def tractor_render_patch(run, camcol, field, band='r'):
    import tractor.sdss as sdss
    tim,tinf = sdss.get_tractor_image_dr8(run, camcol, field, band, psf='kl-gm',
                             roi=[500,600,500,600], nanomaggies=True)
    psf = tim.getPsf()
    print 'PSF', psf
    dx, dy = 0., 0.
    #for i,(dx,dy) in enumerate([
	#    (0.,0.), (0.2,0.), (0.4,0), (0.6,0),
	#    (0., -0.2), (0., -0.4), (0., -0.6)]):
    px,py = 50.+dx, 50.+dy
    patch = psf.getPointSourcePatch(px, py)
    print 'Patch size:', patch.shape
    print 'x0,y0', patch.x0, patch.y0
    H,W = patch.shape
    XX,YY = np.meshgrid(np.arange(W), np.arange(H))
    im = patch.getImage()
    cx = patch.x0 + (XX * im).sum() / im.sum()
    cy = patch.y0 + (YY * im).sum() / im.sum()
    print 'cx,cy', cx,cy
    print 'px,py', px,py

    #self.assertLess(np.abs(cx - px), 0.1)
    #self.assertLess(np.abs(cy - py), 0.1)
   
    plt.clf()
    plt.imshow(patch.getImage(), interpolation='nearest', origin='lower')
    plt.title('dx,dy %f, %f' % (dx,dy))
    plt.savefig('pixpsf-%i.png' % i)


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
    # subselect stripe field 672 - get existing sources
    ########################################################
    run, camcol, field = 4263, 4, 367
    idx = np.where(primary_matched.field == field)[0]
    #primary_field_df = primary_matched[primary_matched.field == field]
    primary_field_df = primary_matched.iloc[idx]
    coadd_field_df   = coadd_matched.iloc[idx]

    # grab fits images 
    imgfits = make_fits_images(run, camcol, field)

    sys.exit()

    #############################################
    # initialize celeste model
    #############################################
    import CelestePy.model_sources as models
    reload(models)
    model = models.CelesteGMMPrior(
            star_flux_prior_distn = None,
            gal_flux_prior_distn  = None,
            # patch epsilon options 
            )

    # for each run/camcol/field, add a data
    model.add_field(img_dict = imgfits)

    # initialize sources from photo obj catalog
    model.initialize_sources(photoobj_df = primary_field_df)

    # look at pixel error and a few plots based on distance and source fluxes
    if False:
        examine_pixel_error(model)
        examine_brightest_sources(model)

    bsrcs, bidx = model.get_brightest(object_type='star', num_srcs=50, return_idx=True)
    bsrcs = bsrcs[10:21]
    bidx  = bidx[10:21]

    ##########################
    # DEBUG
    if False:
        for bright_i, s in enumerate(bsrcs):
            examine_initialization(s, id="bright_%d"%bright_i, imgfits=imgfits, model=model)
            plt.close("all")

        rmod_img = model.render_model_image(imgfits['r'])

    ###END DEBUG #######################

    # breadcrumbs - make sure we can examine which source corresponds to
    # which catalog entry
    blocs = np.array([s.params.u for s in bsrcs])
    plocs = primary_field_df[['ra', 'dec']].values[bidx,:]
    assert np.allclose(blocs, plocs), "not the same location! noooo"


    ######################################
    # gibbs step on a handful of sources #
    ######################################
    # do some resampling, each source keeps each sample
    Nsamps = 20
    for i in pyprind.prog_bar(xrange(Nsamps)):

        # resample photon images
        model.field_list[0].resample_photons(bsrcs)

        # resample source params
        for s in bsrcs:
            s.resample_star()
            #resample_location()
            #s.resample_fluxes()
            s.store_sample()
            s.store_loglike()


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



