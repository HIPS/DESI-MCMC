import os
import sys
from math import pi, sqrt, ceil, floor
from datetime import datetime

import pyfits
import pylab as plt
import numpy as np

from tractor.engine import *
from tractor.basics import *
from tractor.sdss import *

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import setRadecAxes, redgreen
from astrometry.libkd.spherematch import match_radec

import pickle

pkl_file = open('real_data/data_1752_3_164_r.pkl', 'rb')
data = pickle.load(pkl_file)
print data
