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
    primary_matched, coadd_matched, dists = create_matched_dataset(primary_df, coadd_df)

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

    # initialize sources from first tractor source
    model.initialize_sources(init_src_params = [tractor_src_to_celestepy_src(tsrcs[0])])
    src = model.srcs[0] 

    print "======= running celeste sampler ========"
    # do some resampling, each source keeps each sample
    Nsamps = 1
    star_iters = 0
    for i in pyprind.prog_bar(xrange(Nsamps)):
        # sample parameters from data
        src.resample_type()
        if src.is_star():
            star_iters += 1

        # sample data from parameters
        new_images = {}
        for band,image in imgfits.iteritems():
            model_img, xlim, ylim = src.compute_model_patch(img)
            new_images[band] = copy.deepcopy(image)
            new_images[band][ylim[0]:ylim[1]][xlim[0]:xlim[1]] = np.random.poisson(model_img)

        model.field_list = []
        model.add_field(img_dict = new_images)

    print "total iters:", Nsamps
    print "total star iters:", star_iters

