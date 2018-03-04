# -*- coding: utf-8 -*-
"""
Compile cython code.
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="gram_schmidt", ext_modules=cythonize('gram_schmidt.pyx'), include_dirs=[numpy.get_include()])
setup(name="compute_diameter", ext_modules=cythonize('compute_diameter.pyx'), include_dirs=[numpy.get_include()])