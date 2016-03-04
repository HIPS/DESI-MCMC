import numpy as np
import os
import fitsio
import rtree
from astropy import wcs
import pandas as pd


def create_matched_dataset(primary_df, coadd_df):
    """ takes two sets of sources, and matches them up by RA/DEC location """
    # create spatial index with coadd locations
    coadd_lookup = rtree.index.Index()
    for i in xrange(coadd_df.shape[0]):
        ra, dec = coadd_df[['ra', 'dec']].values[i,:]
        coadd_lookup.insert(i, (ra, dec, ra, dec))

    # look through each primary source, find the closest coadd source
    N = primary_df.shape[0]
    good_match = np.ones((N,), dtype=bool)      # is this primary source a match?
    match_idx  = np.zeros((N,), dtype=np.int)   # which coadd_idx is the match?
    dists      = np.zeros((N,))
    for i in xrange(N):
        # find closest coadd to primary_i
        ra, dec      = primary_df[['ra', 'dec']].values[i, :]
        nearest_idxs = list(coadd_lookup.nearest((ra, dec, ra, dec), 2))
        match_idx[i] = nearest_idxs[0]

        # look at the r-band - is it bright enough?
        #prim_r = np.exp(25. - primary_df['psfMag_r'].values[i])
        #coad_r = np.exp(25. - coadd_df['psfMag_r'].values[nearest_idxs[0]])

        # make sure the dist is less that 1e-4 and the second closest is 3 times farther away
        diff1 = np.array([ra, dec]) - coadd_df[['ra', 'dec']].values[nearest_idxs[0],:]
        dist1 = np.sqrt(np.sum(diff1**2))  # dist to closest
        diff2 = np.array([ra, dec]) - coadd_df[['ra', 'dec']].values[nearest_idxs[1],:]
        dist2 = np.sqrt(np.sum(diff2**2))  # dist to second closest
        if dist1 > 1e-4 or (dist2 / dist1) < 3:
            good_match[i] = False
            dists[i] = dist1

    # make matched dataframe 
    coadd_match_df = coadd_df.iloc[match_idx]

    # subset dfs to good matches
    primary_matched = primary_df[ good_match ]
    coadd_matched   = coadd_match_df[ good_match ] 

    return primary_matched, coadd_matched, dists


def df_from_fits(filename, i=1):
    """ create a pandas dataframe from a fits file """
    return pd.DataFrame.from_records(fitsio.FITS(filename)[i].read().byteswap().newbyteorder())


if __name__=="__main__":

    ### params ###
    bands = ['u', 'g', 'r', 'i', 'z']
    primary_run = 6425
    primary_camcol = 4
    primary_fields = range(672, 706)

    ###################################################
    # load in each catalog file as a pandas dataframe #
    ###################################################
    test_primary_fn = "square_4263_4.fit"
    test_coadd_fn   = "square_106_4.fit"
    primary_df      = df_from_fits(test_primary_fn)
    coadd_df        = df_from_fits(test_coadd_fn)

    # create a matched dataset - coadd source (ground truth) to 
    # primary sources (baseline)
    primary_matched, coadd_matched, dists = create_matched_dataset(primary_df, coadd_df)


    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.ion()
    plt.scatter(coadd_matched['psfMag_r'], primary_matched['psfMag_r'])
    plt.errorbar(coadd_matched['psfMag_r'], primary_matched['psfMag_r'],
                 yerr = 2*primary_matched['psfMagErr_r'].values, 
                 linestyle='none')
    ylo, yhi  = np.percentile(primary_matched['psfMag_r'], [1, 99])
    plt.plot([ylo, yhi], [ylo, yhi])
    plt.ylim((ylo, yhi))
    plt.xlim((ylo, yhi))
    plt.xlabel("coadd R mags")
    plt.ylabel("primary R mags")
    plt.show()
    plt.close("all")


    ##### visualize distance #######
    abs_error = np.abs(coadd_matched['psfMag_r'].values - primary_matched['psfMag_r'].values)
    err_order = np.argsort(abs_error)
    plt.scatter(abs_error[err_order], dists[err_order]) 
    plt.ylim((dists.min(), dists.max()))
    plt.xlabel("Absolute R Mag Error")
    plt.ylabel("Distance between coadd center and primary center")
    plt.title("Matched Source Distance and Coadd Error")
    plt.show()


    # look at some statistics of the error
    from autil.stats import make_error_df
    #make_error_df(coadd_matched['psfMag_r'].values, primary_matched['psfMag_r'].values, 
    #              coadd_matched['psfMag_r'].values - coadd_matched['psfMag_)


    gal_df = df_from_fits("/Users/acm/Dropbox/Proj/astro/rnd.astro/existing_catalogs/stars.fits")

