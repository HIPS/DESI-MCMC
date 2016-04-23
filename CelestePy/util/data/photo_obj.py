import pandas as pd
import numpy as np
import fitsio
import rtree


def load_celeste_dataframe(fname):
    """
    Prepare a dataframe 
    load_s82(fname)
    Load Stripe 82 objects into a DataFrame. `fname` should be a FITS file
    created by running a CasJobs (skyserver.sdss.org/casjobs/) query
    on the Stripe82 database. Run the following query in the \"Stripe82\"
    context, then download the table as a FITS file.
    ```
    select
      objid, rerun, run, camcol, field, flags,
      ra, dec, probpsf,
      psfmag_u, psfmag_g, psfmag_r, psfmag_i, psfmag_z,
      devmag_u, devmag_g, devmag_r, devmag_i, devmag_z,
      expmag_u, expmag_g, expmag_r, expmag_i, expmag_z,
      fracdev_r,
      devab_r, expab_r,
      devphi_r, expphi_r,
      devrad_r, exprad_r
    into mydb.s82_0_1_0_1
    from stripe82.photoobj
    where
      run in (106, 206) and
      ra between 0. and 1. and
      dec between 0. and 1.
    ```
    """
    # Read the FITS table into a pandas dataframe
    objs   = df_from_fits(fname)

    # get single set of fluxes for each galaxy, based on majority dev or exp
    usedev = (objs['fracdev_r'] > 0.5).values  # true=> use dev, false=> use exp
    gal_mags = {}
    for b in ['u', 'g', 'r', 'i', 'z']:
        gal_mags[b] = np.where(usedev, objs['devmag_%s'%b].values,
                                       objs['expmag_%s'%b].values)

    # Note that the SDSS photo pipeline doesn't constrain the de Vaucouleur
    # profile parameters and exponential disk parameters (A/B, angle, scale)
    # to be the same, whereas Celeste does. Here, we pick one or the other
    # from SDSS, based on fracdev - we'll get the parameters corresponding
    # to the dominant component. Later, we limit comparison to objects with
    # fracdev close to 0 or 1 to ensure that we're comparing apples to apples.
    gal_ab = np.where(usedev, objs['devab_r'].values, objs['expab_r'].values)

    # gal effective radius (re)
    re_arcsec = np.where(usedev, objs['devrad_r'].values, objs['exprad_r'].values)
    re_pixel  = re_arcsec / 0.396

    # gal angle (degrees)
    raw_phi = np.where(usedev, objs['devphi_r'].values, objs['expphi_r'].values)

    # assemble data frame using celeste parameterization
    result = pd.DataFrame({
        'objid'           : objs['objid'],
        'ra'              : objs['ra'],
        'dec'             : objs['dec'],
        'is_star'         : [p != 0 for p in objs['probpsf']],
        'star_mag_r'      : objs['psfmag_r'],
        'gal_mag_r'       : gal_mags['r'],
        'star_color_ug'   : objs['psfmag_u'] - objs['psfmag_g'],
        'star_color_gr'   : objs['psfmag_g'] - objs['psfmag_r'],
        'star_color_ri'   : objs['psfmag_r'] - objs['psfmag_i'],
        'star_color_iz'   : objs['psfmag_i'] - objs['psfmag_z'],
        'gal_color_ug'    : gal_mags['u'] - gal_mags['g'],
        'gal_color_gr'    : gal_mags['g'] - gal_mags['r'],
        'gal_color_ri'    : gal_mags['r'] - gal_mags['i'],
        'gal_color_iz'    : gal_mags['i'] - gal_mags['z'],
        'gal_fracdev'     : objs['fracdev_r'],
        'gal_ab'          : gal_ab,
        'gal_pixel_scale' : re_pixel,
        'gal_arcsec_scale': re_arcsec,
        'gal_angle_deg'   : raw_phi,
        'run'             : objs['run'],
        'camcol'          : objs['camcol'],
        'field'           : objs['field']
    }).convert_objects(convert_numeric=True)
    return result


def df_from_fits(filename, i=1):
    """ create a pandas dataframe from a fits file """
    return pd.DataFrame.from_records(fitsio.FITS(filename)[i].read().byteswap().newbyteorder())


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



def mags_to_colors(mags, ridx = 2):
    rmag = mags[ridx]
    colors = np.diff(mags[::-1])[::-1]
    return rmag, colors

def colors_to_mags(rmag, colors):
    ug, gr, ri, iz = colors
    g = gr + rmag
    u = ug + g
    i = rmag - ri
    z = i - iz
    return np.array([u, g, rmag, i, z])

colors = ['ug', 'gr', 'ri', 'iz']

# simple converter between mags and nanomaggies
def mags2nanomaggies(mags):
    return np.power(10., (mags - 22.5)/-2.5)

def nanomaggies2mags(nanos):
    return (-2.5)*np.log10(nanos) + 22.5

def celeste_src_to_dict(src):
    """ returns a celeste dataframe row given a celeste source param object """
    star_mag_r, star_colors = mags_to_colors(nanomaggies2mags(src.params.star_fluxes))
    gal_mag_r, gal_colors   = mags_to_colors(nanomaggies2mags(src.params.star_fluxes))
    frac_dev, sigma_rad, phi, ab = src.params.gal_shape
    return {
        'objid'           : src.params.objid,
        'ra'              : src.params.u[0],
        'dec'             : src.params.u[1],
        'is_star'         : src.is_star(),
        'p_star'          : src.params.p_star,
        'star_mag_r'      : star_mag_r,
        'gal_mag_r'       : gal_mag_r,
        'star_color_ug'   : star_colors[0],
        'star_color_gr'   : star_colors[1],
        'star_color_ri'   : star_colors[2],
        'star_color_iz'   : star_colors[3],
        'gal_color_ug'    : gal_colors[0],
        'gal_color_gr'    : gal_colors[1],
        'gal_color_ri'    : gal_colors[2],
        'gal_color_iz'    : gal_colors[3],
        'gal_fracdev'     : frac_dev,
        'gal_ab'          : ab,
        'gal_pixel_scale' : sigma_rad / 0.396,
        'gal_arcsec_scale': sigma_rad,
        'gal_angle_deg'   : phi,
        'run'             : src.params.run,
        'camcol'          : src.params.camcol,
        'field'           : src.params.field
    }




