#! /bin/bash
set -e

mkdir tractor
cd tractor
pwd

git clone https://github.com/dstndstn/tractor.git .
wget "http://astrometry.net/downloads/astrometry.net-0.50.tar.gz"
tar xzf astrometry.net-0.50.tar.gz
mv astrometry.net-0.50 astrometry
(cd astrometry && make pyutil)
(cd astrometry/libkd && make pyspherematch)
(cd astrometry/sdss && make)

make

echo 'Setting up FITSIO:'
mkdir fitsio-git
git clone https://github.com/esheldon/fitsio.git fitsio-git/src
home_dir="$(pwd)/fitsio-git"
echo "Installing FITSIO with home $home_dir"
(cd fitsio-git/src && python setup.py install --home=$home_dir)
mv fitsio-git/lib*/python/fitsio .
