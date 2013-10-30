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
    ccd.regions["data"] = ccd[:,:-64]
    ccd.regions["science"] = ccd[:,100:200]

    # overscan area
    ccd.regions["overscan"] = ccd[:,-64:]

    # create an observing run object, which holds paths and some global things
    #   like the ccd object, maybe Site object?
    path = os.path.join("/Users/adrian/Documents/GraduateSchool/Observing/",
                        "2013-10_MDM")
    obs_run = ObservingRun(path, ccd=ccd)

    utc = Time(datetime(2013,10,25), scale="utc")
    night = ObservingNight(utc=utc, observing_run=obs_run)
    # Standard:
    # obj = TelescopePointing(night,
    #                         files=['m102413.0036.fit','m102413.0037.fit',\
    #                                'm102413.0038.fit'],
    #                         arc_files=['m102413.0039.fit','m102413.0040.fit'])

    # faint rr lyrae:
    # obj = TelescopePointing(night,
    #                         files=['m102413.0050.fit','m102413.0051.fit',\
    #                                'm102413.0052.fit'],
    #                         arc_files=['m102413.0048.fit','m102413.0049.fit',
    #                                    'm102413.0053.fit','m102413.0054.fit'])

    # RR Lyrae itself:
    # obj = TelescopePointing(night,
    #                         files=['m102613.0036.fit','m102613.0037.fit',\
    #                                'm102613.0038.fit'],
    #                         arc_files=['m102613.0039.fit','m102613.0040.fit'])

    # Jules target
    obj = TelescopePointing(night,
                            files=['m102413.0112.fit'],
                            arc_files=['m102413.0113.fit','m102413.0114.fit'])

    arc_file = os.path.join(obs_run.redux_path, "arc", "master_arc.pickle")
    if not os.path.exists(arc_file):
        # - median a bunch of arc images, extract a 1D arc spectrum
        obs_run.make_master_arc(night, narcs=10, overwrite=True)
        arc = obs_run.master_arc

        # fit for a rough wavelength solution
        arc.solve_wavelength(obs_run, find_line_list("Hg Ne"))
        # or, more control:
        # arc._hand_id_lines(Nlines=4)
        # arc._solve_all_lines(line_list, dispersion_fit_order=3)
        # arc._fit_solved_lines(order=5)
        # arc.wavelength = arc.pix_to_wavelength(arc.pix)

        fnpickle(arc, arc_file)

    obs_run.master_arc = fnunpickle(arc_file)

    # TODO: line_ids=True identifies all lines
    fig,ax = obs_run.master_arc.plot(line_ids=True)
    fig.savefig(os.path.join(obs_run.redux_path, "plots", "master_arc.pdf"))
    plt.close()

    # - create a master bias frame. each object will get overscan subtracted,
    #   but this will be used to remove global ccd structure.
    master_bias = night.make_master_bias()

    # make master flat
    master_flat = night.make_master_flat()
    shp = master_flat.shape

    # fit a 2D response function to the flat to smooth over the fringing
    x, y = np.mgrid[:shp[0], :shp[1]]
    p = models.Polynomial2DModel(degree=9)
    fit = fitting.LinearLSQFitter(p)
    fit(x, y, master_flat)
    smooth_flat = p(x,y)

    # TODO: make 2d image plots + 1d trace along a few columns plus fit in those columns
    sanity_check_flat

    # plot the flat / response function
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(p(x,y), aspect='equal', cmap=cm.Greys,
    #            interpolation='nearest')
    # plt.subplot(132)
    # plt.imshow(master_flat, aspect='equal', cmap=cm.Greys,
    #            interpolation='nearest')
    # plt.subplot(133)
    # plt.imshow(master_flat/p(x,y), aspect='equal', cmap=cm.Greys,
    #            interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    for obj in [obj]:
        # obj.arc_files should work?

        for fn in obj.file_paths:

            # subtract bias
            frame_data = fits.getdata(fn)
            frame_data = ccd.bias_correct_frame(frame_data, master_bias)

            #TODO: frame.inverse_variance # automatically created from:
            # variance_image = image_data*gain + read_noise**2

            # divide by master flat
            frame_data /= smooth_flat

            collapsed_spec = np.median(frame_data, axis=0)[100:-100]
            row_pix = np.arange(len(collapsed_spec))
            g = gaussian_fit(row_pix, collapsed_spec,
                             mean=np.argmax(collapsed_spec))

            # plt.plot(row_pix, collapsed_spec, drawstyle='steps', lw=2.)
            # plt.plot(row_pix, g(row_pix), drawstyle='steps')
            # plt.plot(np.arange(0,len(collapsed_spec),0.01),
            #          g(np.arange(0,len(collapsed_spec),0.01)),
            #          drawstyle='steps')
            # plt.axvline(g.mean.value + 5*g.stddev.value)
            # plt.axvline(g.mean.value - 5*g.stddev.value)
            # plt.show()

            L_idx = int(round(g.mean.value - 5*g.stddev.value)) + 100
            R_idx = int(round(g.mean.value + 5*g.stddev.value)) + 100

            spec = np.sum(frame_data[:,L_idx:R_idx], axis=1)
            spec /= float(R_idx-L_idx)

            sky_l = np.median(frame_data[:,:L_idx], axis=1)
            sky_r = np.median(frame_data[:,R_idx:], axis=1)
            sky = 0. #(sky_l + sky_r) / 2.

            plt.plot(obs_run.master_arc.wavelength, spec-sky,
                     drawstyle="steps")
            plt.show()

            return

            # TODO: flag CR's
            #frame.flag_cosmic_rays(obs_run)

            # create 2D wavelength image
            # TODO: cache this!
            wvln_2d = obj.solve_2d_wavelength()

            sky_idx = np.zeros_like(wvln_2d).astype(bool)
            sky_idx[:,150:] = True

            from scipy.interpolate import UnivariateSpline
            s = UnivariateSpline(wvln_2d[sky_idx], frame_data[sky_idx])

            return


            # sky subtract
            frame.sky_subtract(obs_run)

        """
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
        """