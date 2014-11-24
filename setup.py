from distutils.core import setup
from Cython.Build import cythonize
import numpy

#setup(
#   ext_modules=cythonize('**/*.pyx'), 
#   include_dirs=[numpy.get_include(),], 
#)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension(
    "gmm_like_fast",
    ["gmm_like_fast.pyx"],
    extra_compile_args=['-openmp'],
    extra_link_args=['-openmp'],
)

setup(
    name         = 'Guassian Mixture Model Density', 
    cmdclass     = {'build_ext': build_ext},
    ext_modules  = [ext_module],
    include_dirs = [numpy.get_include(),],
)

