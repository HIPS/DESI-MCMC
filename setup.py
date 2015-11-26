from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


celeste_sample_sources_ext = \
    Extension("CelestePy/celeste_sample_sources",
            # ORDER IMPORTANT HERE!
            ['deps/randomkit/randomkit.c', 
             'deps/randomkit/distributions.c',
             'CelestePy/celeste_sample_sources.pyx'],
             include_dirs = ['deps/randomkit', numpy.get_include()],
             language="c++",             # generate C++ code
            )

setup(
   ext_modules  = cythonize(['CelestePy/util/like/*.pyx',
                             'CelestePy/celeste_fast.pyx',
                              celeste_sample_sources_ext]),
   include_dirs = [numpy.get_include(),],
)

