from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Get numpy path
import os, numpy
numpy_base_path = os.path.split(numpy.__file__)[0]
numpy_incl_path = os.path.join(numpy_base_path, "core", "include")

lm10_acc = Extension("streams.potential._lm10_acceleration",
                      ["streams/potential/_lm10_acceleration.pyx"],
                     include_dirs=[numpy_incl_path])
pal5_acc = Extension("streams.potential._pal5_acceleration",
                      ["streams/potential/_pal5_acceleration.pyx"],
                     include_dirs=[numpy_incl_path])

gc_hel = Extension("streams.coordinates._gc_hel",
                   ["streams/coordinates/_gc_hel.pyx"],
                    include_dirs=[numpy_incl_path])

setup(
    name="Streams",
    version="0.0",
    author="Adrian Price-Whelan",
    author_email="adrn@astro.columbia.edu",
    license="BSD",
    cmdclass = {'build_ext': build_ext},
    ext_modules=[lm10_acc, pal5_acc, gc_hel]
)
