#
# Script to download a sample of DR10
#
import fitsio
import numpy as np
import seaborn as sns
from redshift_utils import load_sdss_fluxes_clean_split
import urllib, os, sys

def download_spec_file(plate, mjd, fiberid, redownload=False):
    """ grabs the spec file given plate, mjd and fiber id """
    spec_url_template  = "http://data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/%04d/spec-%04d-%05d-%04d.fits\n"
    spec_url = spec_url_template%(plate, plate, mjd, fiberid)
 
    # check if ../../data/DR10QSO/spec/<FNAME> exists! if so, skip it!
    bname = os.path.basename(spec_url.strip())
    fpath = "../../data/DR10QSO/specs/%s"%bname
    if not redownload and os.path.exists(fpath):
        print "    already there, skipping", bname
        return fpath

    # otherwise, download it
    def dlProgress(count, blockSize, totalSize):
      percent = int(count*blockSize*100/totalSize)
      sys.stdout.write("\r    " + bname + "...%d%%" % (percent))
      sys.stdout.flush()
    urllib.urlretrieve(spec_url.strip(), fpath, reporthook=dlProgress)
    print ""
    return fpath
   

if __name__=="__main__":

    ## scrape values corresponding to sampled quasars
    #qso_sample_files = glob('cache_remote/photo_experiment0/redshift_samples*chain_0.npy')
    qso_sample_files = glob('cache_remote/temper_experiment/redshift_samples*.npy')
    qso_ids = []
    for i in to_inspect:
        _, _, _, qso_info, _ = load_redshift_samples(qso_sample_files[i])
        qso_ids.append([qso_info[b] for b in ['PLATE', 'MJD', 'FIBERID']])

    ## load up the DR10QSO file
    #dr10qso = fitsio.FITS('../../data/DR10QSO/DR10Q_v2.fits')
    #qso_df = dr10qso[1].read()
    #remove those with zwarning nonzero
    #qso_df = qso_df[ qso_df['ZWARNING']==0 ]
    #randomly select 100 quasars
    #Nquasar = len(qso_df)
    #np.random.seed(42)
    #perm    = np.random.permutation(Nquasar)
    #idx     = perm[0:1000]
    
    # get their PLATE-MJD-FIBER, and assemble filenames
    #qso_ids   = qso_df[['PLATE', 'MJD', 'FIBERID']][idx]
    spec_url_template  = "http://data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/%04d/spec-%04d-%05d-%04d.fits\n"
    qso_lines = [spec_url_template%(qid[0], qid[0], qid[1], qid[2]) for qid in qso_ids]

    # write to little file for wget...
    f = open('qso_list.csv', 'w')
    f.writelines(qso_lines)
    f.close()

# zip through and download spec files
    for i, qso_url in enumerate(qso_lines):

        # check if ../../data/DR10QSO/spec/<FNAME> exists! if so, skip it!
        bname = os.path.basename(qso_url.strip())
        fpath = "../../data/DR10QSO/specs/%s"%bname
        if os.path.exists(fpath):
            print "    already there, skipping", bname
            continue

        # otherwise, download it
        def dlProgress(count, blockSize, totalSize):
          percent = int(count*blockSize*100/totalSize)
          sys.stdout.write("\r    " + bname + "...%d%% (%d of %d)" % (percent, i, len(qso_lines)) )
          sys.stdout.flush()
        urllib.urlretrieve(qso_url.strip(), fpath, reporthook=dlProgress)
        print ""


##data.sdss3.org/sas/dr10/boss/spectro/redux/v5_5_12/spectra/
#
##spec_file = "/Users/acm/Downloads/spec-3586-55181-0003.fits"  # DR10 Spec
#spec_file = "/Users/acm/Downloads/spec-0685-52203-0467.fits"   # DR7 spec
#dfh = fitsio.read_header(spec_file)
#df  = fitsio.read(spec_file)
#dfits = fitsio.FITS(spec_file)
#
## get the coadd (what's coadd?) spectra and it's sample locations
#spec = dfits[1]['flux'].read()
#spec_ivar = dfits[1]['ivar'].read()
#spec_model = dfits[1]['model'].read()
#lam  = 10 ** dfits[1]['loglam'].read()
#
## get the red-shift
#z = dfits[2]['Z'].read()[0]
#
################################################################################
#### Spectro Flux Information 
##SPECTROFLUX float32[5]  Spectrum projected onto ugriz filters (nanomaggies)
##SPECTROFLUX_IVAR    float32[5]  Inverse variance of spectrum projected onto ugriz filters (nanomaggies)
##SPECTROSYNFLUX  float32[5]  Best-fit template spectrum projected onto ugriz filters (nanomaggies)
##SPECTROSYNFLUX_IVAR float32[5]  Inverse variance of best-fit template spectrum projected onto ugriz filters (nanomaggies)
##SPECTROSKYFLUX  float32[5]  Sky flux in each of the ugriz imaging filters (nanomaggies)
################################################################################
#sp_flux      = dfits[2]['SPECTROFLUX'].read()
#sp_flux_ivar = dfits[2]['SPECTROFLUX_IVAR'].read()
#sp_mod_flux  = dfits[2]['SPECTROSKYFLUX'].read()
#sp_skyflux   = dfits[2]['SPECTROSKYFLUX'].read()
#psf_flux     = dfits[2]['PSFFLUX'].read()
#
## get the 
#qso_nanomaggies = 10 ** ((qso_mags - 22.5)/-2.5)
#
#psf_mags = dfits[2]['PSFMAG'].read()
#
#10**((psf_mags - 22.5) / -2.5)
#
#
#plt.plot(lam, spec)
#plt.plot(lam, spec_model)
#plt.show()
#
