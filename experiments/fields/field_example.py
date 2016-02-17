import astrometry.util.fits as aufits
import tractor.sdss as sdss
import astrometry.sdss as asdss
from CelestePy.util.bound.bounding_box import get_bounding_boxes_idx
from CelestePy.util.data import make_fits_images, tractor_src_to_celestepy_src
import numpy as np
import matplotlib.pyplot as plt
from CelestePy.celeste import FitsImage
from CelestePy.celeste_galaxy_conditionals import gen_galaxy_psf_image
import pandas as pd

if __name__ == '__main__':

    ##############################################
    # load in a full field and tractor sources   #
    #############################################
    run, camcol, field = 125, 1, 17
    tsrcs              = sdss.get_tractor_sources_dr9(run, camcol, field)
    imgfits            = make_fits_images(run, camcol, field)
    srcs               = [tractor_src_to_celestepy_src(s) for s in tsrcs]

    # stack into array - for comparison
    bands = ['u', 'g', 'r', 'i', 'z']
    flux_array = np.array([s.fluxes for s in srcs])
    tractor_fluxes = pd.DataFrame(flux_array, columns=bands)
    tractor_fluxes.describe()

    #############################################
    # initialize celeste model
    #############################################
    import CelestePy.models as models
    reload(models)
    model = models.Celeste(
            star_flux_prior_distn = None,
            gal_flux_prior_distn  = None,
            # patch epsilon options 
            )

    # for each run/camcol/field, add a data
    model.add_field(img_dict = imgfits)
    model.initialize_sources(init_src_params=srcs)

    #####################
    # single gibbs step #
    #####################
    model.field_list[0].resample_photons(model.srcs)
    model.resample_sources()

    # or equivalently, resample the whole model (photon responsibilitys and
    # then source params)
    #model.resample_model()

    # look at one of the sources in celeste
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    fig, axarr = plt.subplots(1, 2, figsize=(12,6))
    model.srcs[3].plot(imgfits['r'], ax=axarr[0], data_ax=axarr[1])

    #plt.close("all")

