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
from scipy.interpolate import interp1d

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

    # - median a bunch of arc images, extract a 1D arc spectrum from
    #   the first night (arbitrary)
    obs_run.make_master_arc(obs_run.nights.values()[0],
                            narcs=10, overwrite=False)
    arc = obs_run.master_arc
    pix = np.arange(len(arc))

    if arc.wavelength is None:
        # fit for a rough wavelength solution
        arc.solve_wavelength(obs_run, find_line_list("Hg Ne"))

    # plot the arc lamp spectrum with lines identified
    fig,ax = obs_run.master_arc.plot(line_ids=True)
    fig.savefig(os.path.join(obs_run.redux_path,
                             "plots", "master_arc.pdf"))
    plt.clf()

    # - the above, rough wavelength solution is used when no arcs were
    #   taken at a particular pointing, and as intial conditions for the
    #   line positions for fitting to each individual arc

    all_spec = []
    for night in obs_run.nights.values(): #[obs_run.nights["m102213"]]:
        for pointing in night.pointings:

            if pointing.object_name != "RR Lyr":
                continue

            pointing.reduce(overwrite=True)

            science_data = fits.getdata(pointing._data_file_paths.values()[0])
            wvln_2d = pointing.wavelength_image

            collapsed_spec = np.median(science_data, axis=0)
            row_pix = np.arange(len(collapsed_spec))
            g = gaussian_fit(row_pix, collapsed_spec,
                             mean=np.argmax(collapsed_spec))

            # define rough box-car aperture for spectrum
            L_idx = int(np.floor(g.mean.value - 4*g.stddev.value))
            R_idx = int(np.ceil(g.mean.value + 4*g.stddev.value))+1

            spec = np.sum(science_data[:,L_idx:R_idx], axis=1)
            spec /= float(R_idx-L_idx)

            spec_wvln = np.mean(wvln_2d[:,L_idx:R_idx], axis=1)

            all_spec.append((spec_wvln, spec))

            continue

        if len(all_spec) >= 3:
            break

    plt.figure(figsize=(12,6))
    first_w = None
    for wv,fx in all_spec:
        w,f = wv[275:375], fx[275:375]

        if first_w is None:
            first_w = w

        ff = interp1d(w,f,bounds_error=False)
        print(w-first_w)
        # TODO: gaussian fit should allow negative values
        #g = gaussian_fit(w, f, mean=6563., log10_amplitude=1E-1)
        plt.plot(first_w, ff(first_w))

    plt.xlim(6500, 6600)
    plt.show()

    if False:
        # create 2D wavelength image
        # TODO: cache this!
        wvln_2d = obj.solve_2d_wavelength(overwrite=False)
        science_data = frame_data[ccd.regions["science"]]

        ## HACK
        collapsed_spec = np.median(science_data, axis=0)
        row_pix = np.arange(len(collapsed_spec))
        g = gaussian_fit(row_pix, collapsed_spec,
                         mean=np.argmax(collapsed_spec))

        # define rough box-car aperture for spectrum
        L_idx = int(np.floor(g.mean.value - 4*g.stddev.value))
        R_idx = int(np.ceil(g.mean.value + 4*g.stddev.value))+1

        spec = np.sum(science_data[:,L_idx:R_idx], axis=1)
        spec /= float(R_idx-L_idx)

        if hdr["EXPTIME"] > 60:
            sky_l = np.median(science_data[:,L_idx-20:L_idx-10], axis=1)
            sky_r = np.median(science_data[:,R_idx+10:R_idx+20], axis=1)
            sky = (sky_l + sky_r) / 2.

            spec -= sky

        s = Spectrum(obs_run.master_arc.wavelength*u.angstrom,
                     spec)
        fig,ax = s.plot()
        ax.set_title(hdr["OBJECT"])
        fig.savefig("/Users/adrian/Downloads/{0}.pdf".format(hdr["OBJECT"]))
        return
        ## HACK

        # first do it the IRAF way:
        row_pix = np.arange(science_data.shape[1])
        for row in science_data:
            g = gaussian_fit(row_pix, row,
                             mean=np.argmax(row))
            L_idx = int(np.floor(g.mean.value - 4*g.stddev.value))
            R_idx = int(np.ceil(g.mean.value + 4*g.stddev.value))+1

            plt.clf()
            plt.plot(row_pix, row, marker='o', linestyle='none')
            plt.axvline(L_idx)
            plt.axvline(R_idx)
            plt.show()
            return

        collapsed_spec = np.median(science_data, axis=0)
        row_pix = np.arange(len(collapsed_spec))
        g = gaussian_fit(row_pix, collapsed_spec,
                         mean=np.argmax(collapsed_spec))

        # define rough box-car aperture for spectrum
        L_idx = int(np.floor(g.mean.value - 5*g.stddev.value))
        R_idx = int(np.ceil(g.mean.value + 5*g.stddev.value))+1


        # grab 2D sky regions around the aperture
        # sky_l = np.ravel(science_data[:,L_idx-20:L_idx-10])
        # sky_l_wvln = np.ravel(wvln_2d[:,L_idx-20:L_idx-10])
        # sky_r = np.ravel(science_data[:,R_idx+10:R_idx+20])
        # sky_r_wvln = np.ravel(wvln_2d[:,R_idx+10:R_idx+20])

        # # make 1D, oversampled sky spectrum
        # sky_wvln = np.append(sky_l_wvln, sky_r_wvln)
        # idx = np.argsort(sky_wvln)
        # sky_wvln = sky_wvln[idx]
        # sky = np.append(sky_l, sky_r)[idx]

        # from scipy.interpolate import UnivariateSpline
        # interp = UnivariateSpline(sky_wvln, sky, k=3)

        spec_2d = science_data[:,L_idx:R_idx]
        spec_wvln = wvln_2d[:,L_idx:R_idx]
        spec_sky = interp(spec_wvln[:,3])

        plt.plot(spec_wvln[:,3],
                 (spec_2d[:,3] - spec_sky),
                 drawstyle="steps")
        plt.show()
        return

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
    from argparse import ArgumentParser
    import logging

    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main()