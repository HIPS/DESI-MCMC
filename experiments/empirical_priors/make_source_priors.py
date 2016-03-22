import autograd.numpy as np
from CelestePy.util.data import mags2nanomaggies, df_from_fits
from CelestePy.util.dists.mog import MixtureOfGaussians
from CelestePy.util.dists.flux_prior import FluxColorMoG, GalShapeMoG
import cPickle as pickle
import pandas as pd
import pyprind

# cross validate mog function
def fit_mog(data, max_comps = 20, mog_class = MixtureOfGaussians):
    from sklearn import mixture
    N            = data.shape[0]
    train        = data[:int(.75*N), :]
    test         = data[int(.75*N):, :]

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
    shape_df[:,2] = shape_df[:,2] * (np.pi / 180.)
    shape_df = GalShapeMoG.to_unconstrained(shape_df)
    bad_idx = np.any(np.isinf(shape_df), axis=1)
    shape_df = shape_df[~bad_idx,:]
    gal_shape_mog = fit_mog(shape_df, mog_class = GalShapeMoG, max_comps=50)
    with open('gal_shape_mog.pkl', 'wb') as f:
        pickle.dump(gal_shape_mog, f)

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


