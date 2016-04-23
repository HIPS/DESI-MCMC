import autograd.numpy as np
from CelestePy.util.data import mags2nanomaggies, df_from_fits
from CelestePy.util.dists.mog import MixtureOfGaussians
from CelestePy.util.dists.flux_prior import FluxColorMoG, GalShapeMoG, GalRadiusMoG, GalAbMoG
import cPickle as pickle
import pandas as pd
import pyprind
import fitsio

# cross validate mog function
def fit_mog(data, max_comps = 20, mog_class = MixtureOfGaussians):
    from sklearn import mixture
    N            = data.shape[0]
    if len(data.shape) == 1:
        train = data[:int(.75 * N)]
        test  = data[int(.75 * N):]
    else:
        train = data[:int(.75*N), :]
        test  = data[int(.75*N):, :]

    # do train/val GMM fit
    num_comps = np.arange(1, max_comps+1)
    scores    = np.zeros(len(num_comps))
    for i, num_comp in enumerate(num_comps):
        g = mixture.GMM(n_components=num_comp, covariance_type='full')
        g.fit(train)
        logprobs, res = g.score_samples(test)
        scores[i] = np.mean(logprobs)
        print "num_comp = %d (of %d) score = %2.4f"%(num_comp, max_comps, scores[i])
    print "best validation, num_comps = %d"%num_comps[scores.argmax()]

    # fit final model to all data
    g = mixture.GMM(n_components = num_comps[scores.argmax()], covariance_type='full')
    g.fit(data)

    # create my own GMM object - it's better!
    return mog_class(g.means_, g.covars_, g.weights_)


if __name__=="__main__":

    # read galaxy and star from FITS
    print "reading in galaxy and star fluxes"
    gals_df  = df_from_fits('gals.fits')
    stars_df = df_from_fits('stars.fits')

    print "reading in co-added galaxies"
    test_coadd_fn = "../../data/stripe_82_dataset/square_106_4.fit"
    coadd_df = df_from_fits(test_coadd_fn)

    #############################
    # Fit Fluxes and save prior #
    #############################
    # get flux mags and nanomaggies for stars/gals
    bands = ['u', 'g', 'r', 'i', 'z']
    mags_gals   = gals_df[['cmodelmag_%s'%b for b in bands]].values
    mags_stars  = stars_df[['psfmag_%s'%b for b in bands]].values

    # create flux dataset (take log - make sure to remove any infinite values)
    def to_log_nanomaggies(mags):
        fluxes  = np.log(mags2nanomaggies(mags))
        bad_idx = np.any(np.isinf(fluxes), axis=1)
        return fluxes[~bad_idx,:]

    fluxes_gals = pd.DataFrame(to_log_nanomaggies(mags_gals), columns = bands)
    fluxes_stars = pd.DataFrame(to_log_nanomaggies(mags_stars), columns = bands)

    # create colors dataset
    colors_gals  = pd.DataFrame(FluxColorMoG.to_colors(fluxes_gals.values),
                                columns = ['cu', 'cg', 'ci', 'cz', 'r'])

    colors_stars = pd.DataFrame(FluxColorMoG.to_colors(fluxes_stars.values),
                                columns = ['cu', 'cg', 'ci', 'cz', 'r'])

    # fit model to colors and to r band flux (reference band)
    star_flux_mog = fit_mog(colors_stars.values[::1000,:], max_comps = 40, mog_class=FluxColorMoG)
    gal_flux_mog  = fit_mog(colors_gals.values[::1000,:], max_comps = 50, mog_class=FluxColorMoG)

    # save pickle files 
    with open('gal_fluxes_mog.pkl', 'wb') as f:
        pickle.dump(gal_flux_mog, f)

    with open('star_fluxes_mog.pkl', 'wb') as f:
        pickle.dump(star_flux_mog, f)

    #####################################################################
    # fit model to galaxy shape parameters
    # 
    #   re  - [0, infty], transformation log
    #   ab  - [0, 1], transformation log (ab / (1 - ab))
    #   phi - [0, 180], transformation log (phi / (180 - phi))
    #
    ######################################################################
    print "fitting galaxy shape"
    shape_df = np.row_stack([ coadd_df[['expRad_r', 'expAB_r', 'expPhi_r']].values,
                              coadd_df[['deVRad_r', 'deVAB_r', 'deVPhi_r']].values ])[::3,:]
    shape_df[:,0] = np.log(shape_df[:,0])
    shape_df[:,1] = np.log(shape_df[:,1]) - np.log(1.-shape_df[:,1])
    shape_df[:,2] = shape_df[:,2] * (np.pi / 180.)

    bad_idx = np.any(np.isinf(shape_df), axis=1)
    shape_df = shape_df[~bad_idx,:]
    gal_re_mog = fit_mog(shape_df[:,0], mog_class = GalRadiusMoG, max_comps=50)
    gal_ab_mog = fit_mog(shape_df[:,1], mog_class = GalAbMoG, max_comps=50)

    with open('gal_re_mog.pkl', 'wb') as f:
        pickle.dump(gal_re_mog, f)

    with open('gal_ab_mog.pkl', 'wb') as f:
        pickle.dump(gal_ab_mog, f)


    #####################################################################
    # fit star => galaxy proposal distributions
    #
    #   re  - [0, infty], transformation log
    #   ab  - [0, 1], transformation log (ab / (1 - ab))
    #   phi - [0, 180], transformation log (phi / (180 - phi))
    #
    ######################################################################
    import CelestePy.util.data as du
    from sklearn.linear_model import LinearRegression
    coadd_df = du.load_celeste_dataframe("../../data/stripe_82_dataset/coadd_catalog_from_casjobs.fit")

    # make star => radial extent proposal
    star_res = coadd_df.gal_arcsec_scale[ coadd_df.is_star ].values
    star_res = np.clip(star_res, 1e-8, np.inf)
    star_res_proposal = fit_mog(np.log(star_res).reshape((-1,1)), max_comps = 20, mog_class = MixtureOfGaussians)
    with open('star_res_proposal.pkl', 'wb') as f:
        pickle.dump(star_res_proposal, f)

    if False:
        xgrid = np.linspace(np.min(np.log(star_res)), np.max(np.log(star_res)), 100)
        lpdf  = star_res_proposal.logpdf(xgrid.reshape((-1,1)))
        plt.plot(xgrid, np.exp(lpdf))
        plt.hist(np.log(star_res), 25, normed=True)
        plt.hist(np.log(star_res), 25, normed=True, alpha=.24)
        plt.hist(star_res_proposal.rvs(684).flatten(), 25, normed=True, alpha=.24)

    # make star fluxes => gal fluxes for tars
    colors    = ['ug', 'gr', 'ri', 'iz']
    star_mags = np.array([du.colors_to_mags(r, c) 
                  for r, c in zip(coadd_df.star_mag_r.values,
                      coadd_df[['star_color_%s'%c for c in colors]].values)])

    gal_mags  = np.array([du.colors_to_mags(r, c) 
                    for r, c in zip(coadd_df.gal_mag_r.values,
                        coadd_df[['gal_color_%s'%c for c in colors]].values)])

    # look at galaxy fluxes regressed on stars
    x = star_mags[coadd_df.is_star.values]
    y = gal_mags[coadd_df.is_star.values]
    star_mag_model = LinearRegression()
    star_mag_model.fit(x, y)
    star_residuals = star_mag_model.predict(x) - y
    star_mag_model.res_covariance = np.cov(star_residuals.T)
    star_resids    = np.std(star_mag_model.predict(x) - y, axis=0)
    with open('star_mag_proposal.pkl', 'wb') as f:
        pickle.dump(star_mag_model, f)

    for i in xrange(5): 
        plt.scatter(star_mag_model.predict(x)[:,i], y[:,i], label=i, c=sns.color_palette()[i])

    plt.legend()
    plt.show()

    # look at star fluxes regressed on galaxy fluxes
    x = gal_mags[~coadd_df.is_star.values]
    y = star_mags[~coadd_df.is_star.values]
    gal_mag_model = LinearRegression()
    gal_mag_model.fit(x, y)
    gal_residuals = gal_mag_model.predict(x) - y
    gal_mag_model.res_covariance = np.cov(gal_residuals.T)
    with open('gal_mag_proposal.pkl', 'wb') as f:
        pickle.dump(gal_mag_model, f)
    for i in xrange(5):
        plt.scatter(gal_mag_model.predict(x)[:,i], y[:,i], label=i, c=sns.color_palette()[i])

    plt.legend()
    plt.show()





    ######################################
    # load in models and test them       #
    ######################################
    gal_flux_mog = pickle.load(open('gal_fluxes_mog.pkl', 'rb'))
    gal_flux_mog.logpdf(fluxes_gals.values[:100,:])

    from autograd import grad
    grad(lambda th: np.sum(gal_flux_mog.logpdf(th)))(fluxes_gals.values[:10,:])

    ###############################################################
    # Visualize distribution of fluxes, colors, shapes, etc       #
    ###############################################################
    # visualize pairwise distributions
    import matplotlib.pyplot as plt
    plt.ion()
    import seaborn as sns
    shape_df = pd.DataFrame(shape_df, columns=['sigma', 'ab', 'phi'])
    sns.pairplot(shape_df)
    plt.show()

    colors_stars['r'] = fluxes_stars['r']
    sns.pairplot(colors_stars)
    plt.suptitle("Stars")

    fig = plt.figure()
    sns.pairplot(fluxes_gals.iloc[::1000,:])
    plt.suptitle("Gals")
    plt.show()

    sns.jointplot(colors_stars['cg'] - colors_stars['cu'], colors_stars['ci'] - colors_stars['cz'])
    plt.show()

