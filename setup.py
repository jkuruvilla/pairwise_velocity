#To build, run python setup.py build_ext --inplace
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("pairwise_pdf.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=[
        Extension("pairwise_pdf", ["pairwise_pdf.c"],
                  include_dirs=[numpy.get_include()]),
                      ],
)

# Or, if you use cythonize() to make the ext_modules list, # include_dirs can be passed to setup()

