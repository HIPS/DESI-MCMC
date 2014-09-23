DESI-MCMC
=========

MCMC for the Dark Energy Spectroscopic Instrument

Steps Ryan did to get Tractor going.
1. Removed any ambient installations of tractor or astrometry.
2. Created a virtualenv with --system-site-packages.
3. Get Dustin's checkout.sh and run it.
4. Change into tractor directory and python setup.py install.
5. Moved tractor/astrometry into virtualenv site packages.
6. Installed pyfits into virtualenv via pip.

