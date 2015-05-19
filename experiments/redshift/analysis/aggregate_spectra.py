#
# Takes a list of 
#
from redshift_utils import load_sdss_fluxes_clean_split, mags2nanomaggies, load_specs_from_disk
from glob import glob
from os.path import basename, splitext
import fitsio
import numpy as np
import seaborn as sns
sns.set_style("white")
current_palette = sns.color_palette()

if __name__=="__main__":

    # 0. load fluxes from DR10QSO file
    dr10qso = fitsio.FITS('../../data/DR10QSO/DR10Q_v2.fits')
    qso_df  = dr10qso[1].read()
    qso_df  = qso_df[ qso_df['ZWARNING']==0 ]   #remove zwarning nonzero ones
    Nquasar = len(qso_df)

    mag_fields = ['UMAG', 'GMAG', 'RMAG', 'IMAG', 'ZMAG']
    mag_errs   = [m + 'ERR' for m in mag_fields]
    qso_nanos  = qso_df['PSFFLUX']
    qso_nanos_ivar = qso_df['IVAR_PSFFLUX']

    # get their PLATE-MJD-FIBER, and assemble filenames
    qso_ids   = qso_df[['PLATE', 'MJD', 'FIBERID']]
    spec_url_template  = "http://data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/%04d/spec-%04d-%05d-%04d.fits\n"
    qso_lines = [spec_url_template%(qid[0], qid[0], qid[1], qid[2]) for qid in qso_ids]
    qso_strings = [ "spec-%04d-%05d-%04d"%(qid[0], qid[1], qid[2]) for qid in qso_ids]


    ## load single spec file
    spec_file = glob("../../data/DR10QSO/specs/spec-*-*-*.fits")[1]
    buff, plate, mjd, fiber = basename(splitext(spec_file)[0]).split('-')
    sdf = fitsio.FITS(spec_file)
    spec_flux = sdf[1]['flux'].read()
    spec_ivar = sdf[1]['ivar'].read()
    spec_lam  = np.power(10., sdf[1]['loglam'].read())
    spec_mod  = sdf[1]['model'].read()

    # 
    spectro_flux = sdf[2]['SPECTROFLUX'].read()
    spectro_syn_flux = sdf[2]['SPECTROSYNFLUX'].read()
    spectro_syn_flux_ivar = sdf[2]['SPECTROSYNFLUX_IVAR'].read()
    psf_flux = sdf[2]['PSFFLUX'].read()
    psf_flux_ivar = sdf[2]['PSFFLUX_IVAR'].read()
    model_flux = sdf[2]['MODELFLUX'].read()

    # cross ref w/ DR10
    qso_idx = qso_strings.index(basename(splitext(spec_file)[0]))
    print qso_df['TARGET_FLUX'][qso_idx]
    print qso_df['PSFFLUX'][qso_idx]

    # cross reference w/ dr7
    sdf7 = fitsio.FITS("../../data/data_for_priors/dr7qso.fit")
    dr7_id = (qso_df['PLATE_DR7'][qso_idx],
              qso_df['MJD_DR7'][qso_idx],
              qso_df['FIBERID_DR7'][qso_idx])

    qso7ids = sdf7[1][['PLATE', 'SMJD', 'FIBER']].read()
    dr7_strings = [ "spec-%04d-%05d-%04d"%(q[0], q[1], q[2]) for q in qso7ids]
    dr7_idx = dr7_strings.index('spec-%04d-%05d-%04d'%(dr7_id[1], dr7_id[0], dr7_id[2]))

    dr7_mags = sdf7[1][mag_fields].read()[dr7_idx]
    dr7_nanos = [mags2nanomaggies(m) for m in dr7_mags]
    print dr7_nanos


    # PLOT Comparison of PSFFLUX and SPECTROSYN_FLUX and their variances
    # for the five bands
    ugriz = ['u', 'g', 'r', 'i', 'z']
    xs = np.arange(5)
    plt.bar(xs, spectro_syn_flux, yerr = 2*np.sqrt(1./spectro_syn_flux_ivar), 
            alpha=.4, width=.4, color=sns.color_palette()[0], label='spectro_flux')
    plt.bar(xs+.5, psf_flux, alpha=.4, width=.4, yerr = 2*np.sqrt(1./psf_flux_ivar),
            color=sns.color_palette()[1], label='PSFFLUX')
    plt.legend()
    plt.savefig('../../tex/quasar_z/figs/psfflux_vs_spectroflux.pdf', bbox_inches='tight')


    ## plot double
    spec_files = glob("../../data/DR10QSO/specs/spec-*-*-*.fits")[4:6]
    fig, axarr = plt.subplots(len(spec_files), 1)
    for i, spec_file in enumerate(spec_files):
        sdf = fitsio.FITS(spec_file)
        spec_flux = sdf[2]['PSFFLUX'].read()
        spec_ivar = sdf[2]['PSFFLUX_IVAR'].read()

        xs = np.arange(5)
        axarr[i].bar(xs, spec_flux, alpha=.4, width=.8, yerr = 2*np.sqrt(1./spec_ivar),
                     error_kw = {'linewidth':5},
                     color=sns.color_palette()[1], label='PSFFLUX')
        axarr[i].set_xticklabels
    plt.legend()
    plt.tight_layout()








