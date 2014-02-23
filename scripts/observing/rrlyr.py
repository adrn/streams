# coding: utf-8
"""
    Test observing classes
"""

from __future__ import absolute_import, unicode_literals, \
                       division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import pytest

# Third-party
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from streams.reduction.observing import *
from streams.reduction.util import *

def main():

    # define the ccd and geometry
    # TODO: units for gain / read_noise?
    ccd = CCD(gain=3.7, read_noise=5.33,
              shape=(1024,364), dispersion_axis=0) # shape=(nrows, ncols)

    # define regions of the detector
    ccd.regions["data"] = ccd[:,:-64]
    ccd.regions["science"] = ccd[:,100:200]
    ccd.regions["overscan"] = ccd[:,-64:]

    # create an observing run object, which holds paths and some global things
    #   like the ccd object, maybe Site object?
    path = os.path.join("/Users/adrian/Documents/GraduateSchool/Observing/",
                        "2013-10_MDM")
    obs_run = ObservingRun(path, ccd=ccd)

    rrlyrs = []
    for subdir in glob.glob(os.path.join(obs_run.redux_path, "m*")):
        for fn in glob.glob(os.path.join(subdir, "*.fit*"):
            hdr = fits.getheader(fn)
            if hdr["OBJECT"] == "RR Lyr":
                rrlyrs.append(fn)

    collapsed_spec = np.median(science_data, axis=0)
    row_pix = np.arange(len(collapsed_spec))
    g = gaussian_fit(row_pix, collapsed_spec,
                     mean=np.argmax(collapsed_spec))

    # define rough box-car aperture for spectrum
    L_idx = int(np.floor(g.mean.value - 4*g.stddev.value))
    R_idx = int(np.ceil(g.mean.value + 4*g.stddev.value))+1

    spec_2d = science_data[:,L_idx:R_idx]

if __name__ == "__main__":
    main()