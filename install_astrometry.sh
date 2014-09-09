#!/bin/bash

set -e

PYTHON_SITE=`python -c "import site; print site.getsitepackages()[0]"`

svn co -r 24361 http://astrometry.net/svn/trunk/src/astrometry

(cd astrometry && make -C sdss)
(cd astrometry && make -C libkd)
(cd astrometry && make -C libkd pyspherematch)

echo "Linking astrometry to your site packages in: $PYTHON_SITE"
ln -fs "$PWD/astrometry" "$PYTHON_SITE/astrometry"

echo "Python will now load astrometry from either the current directory or from:"
(cd / && python -c "import astrometry; print astrometry.__file__")
