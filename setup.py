from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Get numpy path
import os, numpy
numpy_base_path = os.path.split(numpy.__file__)[0]
numpy_incl_path = os.path.join(numpy_base_path, "core", "include")

gc_hel = Extension("streams.coordinates._gc_hel",
                   ["streams/coordinates/_gc_hel.pyx"],
                    include_dirs=[numpy_incl_path])

rewinder = Extension("streams.inference.rewinder_likelihood",
                     ["streams/inference/rewinder_likelihood.pyx"],
                     include_dirs=[numpy_incl_path])

potential = Extension("streams.potential.basepotential",
                      ["streams/potential/basepotential.pyx"],
                      include_dirs=[numpy_incl_path])

lm10 = Extension("streams.potential.lm10",
                 ["streams/potential/lm10.pyx"],
                 include_dirs=[numpy_incl_path])

setup(
    name="Streams",
    version="0.0",
    author="Adrian Price-Whelan",
    author_email="adrn@astro.columbia.edu",
    license="BSD",
    cmdclass = {'build_ext': build_ext},
    ext_modules=[gc_hel, rewinder, potential, lm10]
)
