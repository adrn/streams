from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Get numpy path
import os, numpy
numpy_base_path = os.path.split(numpy.__file__)[0]
numpy_incl_path = os.path.join(numpy_base_path, "core", "include")

gal_hel = Extension("streams.coordinates.gal_hel",
                    ["streams/coordinates/gal_hel.pyx"],
                    include_dirs=[numpy_incl_path])

rewinder = Extension("streams.rewinder.likelihood",
                     ["streams/rewinder/likelihood.pyx"],
                     include_dirs=[numpy_incl_path])

setup(
    name="Streams",
    version="0.0",
    author="Adrian Price-Whelan",
    author_email="adrn@astro.columbia.edu",
    license="MIT",
    cmdclass = {'build_ext': build_ext},
    ext_modules=[gal_hel, rewinder]
)
