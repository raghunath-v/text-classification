"""
#setup_util.py
#To run: setup_util.py build_ext --inplace

from distutils.core import setup, Extension
from Cython.Build import cythonize


import numpy

modules = [
    Extension("kernels", sources=["kernels_c.pyx"])
]

setup(name='kernels', ext_modules=[modules], py_modules=["kernels"], include_dirs=[numpy.get_include()])
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension("kernels_c", ["kernels_c.pyx"])
]

setup(
    # ext_modules=cythonize("ssk_kernel_c.pyx"),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
