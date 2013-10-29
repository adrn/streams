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
from datetime import datetime

# Third-party
import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt

from ..observing import *
from ..util import *

plot_path = "plots/tests/reduction"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_api():

    # define the ccd and geometry
    # TODO: units for gain / read_noise?
    ccd = CCD(gain=3.7, read_noise=5.33,
              shape=(1024,364), dispersion_axis=0) # shape=(nrows, ncols)

    # region of the detector read out
    ccd.regions["data"] = ccd[:,100:200]

    # overscan area
    ccd.regions["overscan"] = ccd[:,-64:]

    # create an observing run object, which holds paths and some global things
    #   like the ccd object, maybe Site object?
    path = os.path.join("/Users/adrian/Documents/GraduateSchool/Observing/",
                        "2013-10_MDM")
    obs_run = ObservingRun(path, ccd=ccd)

    utc = Time(datetime(2013,10,28), scale="utc")
    night = ObservingNight(utc=utc, observing_run=obs_run)

    # - median a bunch of arc images, extract a 1D arc spectrum
    pix,arc = obs_run.make_master_arc(night, narcs=10, overwrite=True)

    # TODO: wait to plot until lines are ID'd
    fig,ax = plot_spectrum(pix, arc)
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Raw counts")
    fig.savefig(os.path.join(obs_run.redux_path, "plots", "master_arc.pdf"))
    plt.close()

    # - have the user hand identify lines on the master arc (if this is
    #   already done, this just reads a cached JSON file)
    hand_id_pix, hand_id_wvln = obs_run.hand_id_lines(night,
                                                      Nlines=4,
                                                      overwrite=False)

    # - now we want to fit all lines to get line wavelengths vs. line pixels.
    #   these values are used as initial conditions for doing 2D
    #   wavelength calibration. again, this is cached so should only be
    #   done once.
    all_line_pix, all_line_wvln = obs_run.solve_all_lines(night,
                                                          line_list("Hg Ne"),
                                                          overwrite=False)

    return

    # polynomial fit to the grid of pixels / wavelengths for the line centers
    pix2wvln, wvln2pix = obs_run.wavelength_solution_1d(all_line_pix,
                                                        all_line_wvln,
                                                        order=3,
                                                        plot=True)

    # - create a master bias frame. each object will get overscan subtracted,
    #   but this will be used to remove global ccd structure.
    # TODO: what parameters?
    obs_run.make_master_bias()

    # make master flat
    # TODO: what parameters?
    obs_run.make_master_flat()

    # TODO: need some way to specify arcs for each object...
    # TODO: maybe each frame is bound to an ObservingRun so I don't have to
    #       keep passing in obs_run below?
    for obj in all_objects:
        obj.arcs # ??

        for frame in obj.frames:
            # subtract bias
            frame.subtract_bias(obs_run)

            # subtract overscan
            frame.subtract_overscan(obs_run)

            frame.inverse_variance # automatically created from:
            # variance_image = image_data*gain + read_noise**2

            # divide by master flat
            frame.divide_flat(obs_run)

            # flag CR's
            frame.flag_cosmic_rays(obs_run)

            # create 2D wavelength image
            frame.solve_wavelength_2d(obs_run)

            # sky subtract
            frame.sky_subtract(obs_run)