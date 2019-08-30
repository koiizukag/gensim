from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension("poincare", sources=["poincare.cpp"], include_dirs=['.', numpy.get_include()], language='c++')
setup(name="poincare", ext_modules=cythonize([ext]))
