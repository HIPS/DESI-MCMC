import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
plt.ion()
from CelestePy.util.data import make_fits_images, tractor_src_to_celestepy_src
import numpy as np
import pandas as pd
import pyprind
import copy

#stripe 82 files
STRIPE_82_DATA_DIR = "../../data/stripe_82_dataset/"
import sys, os
sys.path.append(STRIPE_82_DATA_DIR)
from load_stripe82_square import df_from_fits, create_matched_dataset

def get_active_sources(source, source_list, image):
    """Given an initial source, "source" with a bounding box, find all
    sources in source_list where their bounding box intersects with "source"'s
    bounding box.

    Collect all sources that contribute to this source's background model image
    """
    def intersect(sa, sb, image):
        xlima, ylima = sa.bounding_boxes[image]
        xlimb, ylimb = sb.bounding_boxes[image]
        widtha, heighta = xlima[1] - xlima[0], ylima[1] - ylima[0]
        widthb, heightb = xlimb[1] - xlimb[0], ylimb[1] - ylimb[0]
        return (np.abs(xlima[0] - xlimb[0])*2 < (widtha + widthb)) and \
               (np.abs(ylima[0] - ylimb[0])*2 < (heighta + heightb))
    return [s for s in source_list if intersect(s, source, image) and s is not source]

def generate_background_patch(source, source_list, image):
    active_sources = get_active_sources(source, source_list, image)
    xlim, ylim     = source.bounding_boxes[image]
    if len(active_sources) < 1:
        return image.epsilon * np.ones((ylim[1]-ylim[0], xlim[1]-xlim[0]))
    background     = np.sum([s.compute_model_patch(image, xlim=xlim, ylim=ylim)[0]
                             for s in active_sources], axis=0) + image.epsilon
    return background

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
    model.initialize_sources(photoobj_df = primary_field_df)

    # TODO incorporate 
    from CelestePy.sources import make_bbox_dict
    for s in model.srcs:
        s.bounding_boxes = make_bbox_dict(s.params, imgfits.values(), pixel_radius=30)

    # add background image to src
    src = model.srcs[0]
    src.background_image_dict = {img: generate_background_patch(src, model.srcs, img)
                                 for img in imgfits.values()}

    print "======= running celeste sampler ========"
    # do some resampling, each source keeps each sample
    Nsamps = 500
    star_iters = 0
    src_type   = []
    for i in pyprind.prog_bar(xrange(Nsamps)):
        # sample parameters from data
        src.resample_type(proposal_fun = src.linear_propose_other_type)
        src.resample_fluxes()
        src.resample_shape()

        src_type.append(src.params.a)
        if src.is_star():
            star_iters += 1

        # sample data from parameters
        new_images = {}
        for band,image in imgfits.iteritems():
            model_img, ylim, xlim = src.compute_model_patch(image)
            new_images[band] = copy.deepcopy(image)
            zpois = np.random.poisson(model_img)
            #new_images[band].nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]] = zpois
            image.nelec.flags['WRITEABLE'] = True
            image.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]] = zpois

        print "iter %d, nstars = %d"%(i, np.sum(np.array(src_type)==0))

    print "total iters:", Nsamps
    print "total star iters:", star_iters

