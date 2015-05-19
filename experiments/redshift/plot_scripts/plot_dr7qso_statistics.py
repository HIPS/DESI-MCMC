import numpy as np
from numpy.random import randn
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from redshift_utils import load_sdss_fluxes_clean_split

sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})
np.random.seed(9221999)


if __name__=="__main__":

    # load DR10Q
    qso_df   = fitsio.FITS('../../data/DR10QSO/DR10Q_v2.fits')[1].read()
    r_fluxes = qso_df['PSFFLUX'][:, 2]
    all_zs   = qso_df['Z_VI']


    ## plot five PSF fluxes


    # set TEST INDICES 
    npr.seed(13)
    test_idx       = npr.permutation(len(qso_df))
    high_shift_idx = np.where(all_zs > 4.08557)[0]



#    trainObj, testObj = load_sdss_fluxes_clean_split(Ntest=50000, seed=42)
#
#
#    # check out some color statistics
#    fluxes = trainObj['sdss_fluxes']
#    col_x  = trainObj['sdss_mags'][:,0] - trainObj['sdss_mags'][:,1]
#    col_y  = trainObj['sdss_mags'][:,2] - trainObj['sdss_mags'][:,3]
#
#
#    ## plot KDE estimate of colors
#    with sns.axes_style("white"):
#        idx = np.arange(0, col_ratios.shape[0], 10)
#        sns.jointplot(-col_x[idx], -col_y[idx], kind="kde")
#    plt.show()
#
#
#data.sdss3.org/sas/dr10/boss/spectro/redux/v5_5_12/spectra/


