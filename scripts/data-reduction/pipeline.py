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

    utc = Time(datetime(2013,10,25), scale="utc")
    night = ObservingNight(utc=utc, observing_run=obs_run)

    # Try first with a standard:
    obj = TelescopePointing(night,
                            files=['m102413.0036.fit','m102413.0037.fit',\
                                   'm102413.0038.fit'],
                            arc_files=['m102413.0039.fit','m102413.0040.fit'])
    objects = [obj]

    # - median a bunch of arc images, extract a 1D arc spectrum
    obs_run.make_master_arc(night, narcs=10, overwrite=False)
    arc = obs_run.master_arc

    if arc.wavelength is None:
        # fit for a rough wavelength solution
        arc.solve_wavelength(obs_run, find_line_list("Hg Ne"))
        # or, for more control:
        # arc._hand_id_lines(Nlines=4)
        # arc._solve_all_lines(line_list, dispersion_fit_order=3)
        # arc._fit_solved_lines(order=5)
        # arc.wavelength = arc.pix_to_wavelength(arc.pix)

    # TODO: line_ids=True identifies all lines
    fig,ax = obs_run.master_arc.plot(line_ids=True)
    plt.show()
    return
    fig.savefig(os.path.join(obs_run.redux_path, "plots", "master_arc.pdf"))
    plt.clf()

    return

    # - create a master bias frame. each object will get overscan subtracted,
    #   but this will be used to remove global ccd structure.
    master_bias = night.make_master_bias(overwrite=False)

    # make master flat
    master_flat = night.make_master_flat(overwrite=False)
    shp = master_flat.shape

    master_flat = np.median(master_flat, axis=1)
    p = models.Polynomial1DModel(degree=15)
    fit = fitting.NonLinearLSQFitter(p)

    pix = np.arange(len(master_flat))
    fit(pix, master_flat)
    smooth_flat = p(pix)
    nflat = master_flat / smooth_flat
    nflat = nflat[:,np.newaxis]

    # Validate against Ally's flat
    ally_flat = fits.getdata("/Users/adrian/Downloads/Flat1.fits")
    ally_nflat = fits.getdata("/Users/adrian/Downloads/nFlat1.fits")

    # plt.figure()
    # plt.subplot(311)
    # plt.plot(master_flat)
    # plt.plot(smooth_flat)

    # plt.subplot(312)
    # plt.plot(master_flat/smooth_flat)
    # plt.ylim(0.96, 1.1)

    # plt.subplot(313)
    # plt.plot(flat_allynflat)
    # plt.ylim(0.96, 1.1)

    # plt.show()

    # smooth_flat = smooth_flat[:,np.newaxis] / smooth_flat.max()

    # print(np.mean(master_flat/smooth_flat), np.median(master_flat/smooth_flat))
    # print(np.mean(flat_allynflat), np.median(flat_allynflat))

    # I can't get this to work...
    # TODO: fit a 2D response function to the flat to smooth over the fringing
    # x, y = np.mgrid[:shp[0], :shp[1]]
    # p = models.Polynomial2DModel(degree=13)
    # fit = fitting.LinearLSQFitter(p)
    # fit(x, y, master_flat)
    # smooth_flat = p(x,y)

    # col = 200
    # plt.subplot(411)
    # plt.plot(smooth_flat[:,col])
    # plt.subplot(412)
    # plt.plot(master_flat[:,col]/smooth_flat[:,col])
    # plt.subplot(413)
    # plt.plot(ally_flat[:,col])
    # plt.subplot(414)
    # plt.plot(ally_nflat[:,col])
    # plt.show()

    for obj in [obj]:
        # obj.arc_files should work?

        for fn in obj.file_paths:

            # subtract bias
            frame_data = fits.getdata(fn)
            frame_data = ccd.bias_correct_frame(frame_data, master_bias)

            #TODO: frame.inverse_variance # automatically created from:
            # variance_image = image_data*gain + read_noise**2

            frame_data /= ally_nflat[:,:300]
            #frame_data /= smooth_flat[:,np.newaxis]

            # divide by master flat
            #frame_data /= smooth_flat

            # TODO: flag CR's
            #frame.flag_cosmic_rays(obs_run)

            # create 2D wavelength image
            # TODO: cache this!
            wvln_2d = obj.solve_2d_wavelength(overwrite=False)
            science_data = frame_data[ccd.regions["science"]]

            collapsed_spec = np.median(science_data, axis=0)
            row_pix = np.arange(len(collapsed_spec))
            g = gaussian_fit(row_pix, collapsed_spec,
                             mean=np.argmax(collapsed_spec))

            L_idx = int(np.floor(g.mean.value - 5*g.stddev.value))
            R_idx = int(np.ceil(g.mean.value + 5*g.stddev.value))+1

            # plt.figure()
            # plt.imshow(science_data)
            # plt.show()

            # plt.figure()
            # plt.plot(collapsed_spec)
            # plt.plot(row_pix, g(row_pix))
            # plt.axvline(L_idx)
            # plt.axvline(R_idx)
            # plt.show()

            sky_l = np.median(science_data[:,L_idx-20:L_idx-10], axis=1)
            sky_r = np.median(science_data[:,R_idx+10:R_idx+20], axis=1)
            sky = (sky_l + sky_r) / 2.

            spec = np.sum(science_data[:,L_idx:R_idx], axis=1)
            spec /= float(R_idx-L_idx)

            plt.figure()
            plt.subplot(211)
            plt.title("sky")
            plt.plot(obs_run.master_arc.wavelength, sky,
                     alpha=0.5, lw=2, drawstyle='steps')
            plt.subplot(212)
            plt.title("spec")
            plt.plot(obs_run.master_arc.wavelength, spec,
                     alpha=0.5, lw=2, drawstyle='steps')

            plt.figure()
            plt.plot(obs_run.master_arc.wavelength, spec-sky,
                     alpha=1., lw=1, drawstyle='steps')

            plt.show()
            return

            #from scipy.interpolate import LSQBivariateSpline
            #s = UnivariateSpline(wvln_2d[sky_idx], frame_data[sky_idx])

            plt.plot(obs_run.master_arc.wavelength, spec-sky,
                     drawstyle="steps")
            plt.show()

            return


            # sky subtract
            frame.sky_subtract(obs_run)

if __name__ == "__main__":
    main()