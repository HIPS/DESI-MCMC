import sys, os
import CelestePy.util.data as data_util
import numpy as np
import pandas as pd
import pyprind

#stripe 82 files
STRIPE_82_DATA_DIR = "../../data/stripe_82_dataset/"
#sys.path.append(STRIPE_82_DATA_DIR)
#from load_stripe82_square import df_from_fits, create_matched_dataset

if __name__ == '__main__':

    ###################################################
    # load in each catalog file as a pandas dataframe #
    ###################################################
    test_primary_fn = os.path.join(STRIPE_82_DATA_DIR, "square_4263_4.fit")
    test_coadd_fn   = os.path.join(STRIPE_82_DATA_DIR, "square_106_4.fit")
    #test_coadd_fn   = os.path.join(STRIPE_82_DATA_DIR, "coadd_catalog_from_casjobs.fit")

    # get celeste-parameterized data frames
    primary_df = data_util.df_from_fits(test_primary_fn)
    coadd_df   = data_util.df_from_fits(test_coadd_fn)

    # create a matched dataset - coadd source (ground truth) to 
    # primary sources (baseline)
    primary_mdf, coadd_mdf, dists = \
        data_util.create_matched_dataset(primary_df, coadd_df)

    #################################################
    # look at the breakdown by field
    #################################################
    print "Stripe 82 dataset statistics:"
    field_info = primary_mdf[['run', 'camcol', 'field']].drop_duplicates()
    for field in np.sort(field_info.field):
        primary_field_df = primary_mdf[primary_mdf.field == field]
        num_stars        = np.sum(primary_field_df.is_star)
        num_gals         = np.sum(primary_field_df.is_star)
        print "    field %d: %d stars, %d galaxies (%d total)" % \
                (field, num_stars, num_gals, primary_field_df.shape[0])

    ########################################################
    # subselect stripe field 367 - get existing sources
    ########################################################
    run, camcol, field = 4263, 4, 367
    idx              = np.where(primary_mdf.field == field)[0]
    primary_field_df = primary_mdf.iloc[idx]
    coadd_field_df   = coadd_mdf.iloc[idx]

    #from tractor import sdss as st
    tsrcs = st.get_tractor_sources_dr9(run, camcol, field)

    # grab fits images 
    imgfits = data_util.make_fits_images(run, camcol, field)

    #############################################
    # initialize celeste model
    #############################################
    import CelestePy.model_sources as models
    reload(models)
    model = models.CelesteGMMPrior()

    # for each run/camcol/field, add a data
    model.add_field(img_dict = imgfits)

    # initialize sources from photo obj catalog
    model.initialize_sources(photoobj_df = primary_field_df)
    #model.initialize_sources(init_src_params = [tractor_src_to_celestepy_src(s) for s in tsrcs])

    # get brightest-ish sources
    ssrcs, sidx = model.get_brightest(object_type='star', num_srcs=40, return_idx=True)
    gsrcs, gidx = model.get_brightest(object_type='galaxy', num_srcs=40, return_idx=True)
    bsrcs = ssrcs[38:39] + gsrcs[38:39]
    bidx  = np.concatenate([sidx[38:39], gidx[38:39]])

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
    Nsamps = 10
    for i in pyprind.prog_bar(xrange(Nsamps)):
        # resample photon images
        model.field_list[0].resample_photons(bsrcs, verbose=True)
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

    ########################################
    # create output catalog dataframe
    ########################################
    celeste_df = data_util.output_df(model.srcs)

    ##########################
    # DEBUG
    # look at pixel error and a few plots based on distance and source fluxes
    if False:
        # resample one star and one galaxy
        fig, axarr = plt.subplots(2, 3)
        star = bsrcs[0]
        gal  = bsrcs[-1]
        star.plot(imgfits['r'], *axarr[0])
        #gal.resample()
        gal.plot(imgfits['r'], *axarr[1])
        #t_us = np.array([ np.array(ts.getPosition()) for ts in tsrcs])
        #dists = np.sum((t_us - gal.params.u)**2, axis=1)
        #ts = tsrcs[np.argmin(dists)]
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



