# coding: utf-8

"""
Comp lamps
==========

Requires:

 * data (FITS)
 * line list of all lines to fit

Procedure:

 * Read config to figure out CCD geometry (sub-array)
 * Read in 10 COMP frames
 * Median the 10 frames
 * Write this out to master_arc.fits
 * Slice out a section along the slit axis from the center of the CCD (+/- 50 from center?)
    * Take median along slit axis (for modspec, 'x' direction)
 * Make a plot
 * Fit brightest 4 lines with Gaussians, show to user & ask for identification
 * Fit a polynomial to x = wavelength, y = pixels
 * Use polynomial to predict locations of all lines in linelist
 * Fit sum of Gaussians, initialized at predicted locations

 * Then, generate 2D images of wavelength values for each COMP
 * Write these to arc/arc_m****13.0***.fit

 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import json
import logging

# Third-party
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.reduction import obs_path, hand_id_lines, median_arc
from streams.reduction.util import *

def main():
    run = "2013-10_MDM"
    night = "m102413"
    hg_ne_lines = line_list("HgNe")

    plot_path = os.path.join(obs_path, "reduction", "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    data_path = os.path.join(obs_path, run, "data", night)
    hand_id_file = os.path.join(data_path, "hand_id_{0}.json".format(night))

    pix, arc_1d = median_arc(data_path)
    if not os.path.exists(hand_id_file):
        line_pixels, line_wavelengths = hand_id_lines(pix, arc_1d, \
                                                      plot_path=plot_path)

        with open(hand_id_file, 'w') as f:
            s = dict()
            s["wavelength"] = line_wavelengths
            s["pixel"] = line_pixels
            f.write(json.dumps(s))

    with open(hand_id_file) as f:
        s = json.loads(f.read())
        line_pixels = s["pixel"]
        line_wavelengths = s["wavelength"]

    p = polynomial_fit(line_wavelengths, line_pixels)

    plt.plot(line_wavelengths, line_pixels, marker='o', linestyle='none')
    grid = np.linspace(np.min(line_wavelengths),
                       np.max(line_wavelengths), 100)
    plt.plot(grid, p(grid), linestyle='--', alpha=0.5, color='b')
    plt.savefig(os.path.join(plot_path, "rough_polynomial_fit.png"))

    predicted_pix = p(hg_ne_lines)

    # only get ones within our pixel range
    idx = (predicted_pix>0) & (predicted_pix<1024)
    arc_lines = hg_ne_lines[idx]
    predicted_pix = predicted_pix[idx]

    # sort by increasing pixel values
    sort_idx = np.argsort(predicted_pix)
    arc_lines = arc_lines[sort_idx]
    predicted_pix = predicted_pix[sort_idx]

    # label the lines, IRAF style
    fig,ax = plt.subplots(1,1,figsize=(11,8))
    ax.plot(pix, arc_1d, drawstyle="steps", c='k')

    # fit a gaussian to each line, determine center
    fit_pix = []
    fit_lines = []
    ylim_max = 0.
    for c_pix,wvln in zip(predicted_pix, arc_lines):
        c_idx = int(c_pix)

        line_pix = pix[c_idx-5:c_idx+5]
        line_data = arc_1d[c_idx-5:c_idx+5]
        p_opt = gaussian_fit(line_pix, line_data)
        c, log_amplitude, stddev, line_center = p_opt

        if abs(line_center - c_pix) > 1.:
            logger.info("Line {0} fit failed.".format(wvln))
            continue

        fit_pix.append(line_center)

        y = max(spectral_line_model(p_opt, line_pix)) + 3000.
        ylim_max = max(ylim_max, y)
        ax.text(line_center-6, y, "{0:.3f}".format(wvln),
                rotation=90, fontsize=10)

        # TODO: if plot, make each individual line plot?
        # l_fig, l_ax = plt.subplots(1,1,figsize=(4,4))
        # l_ax.plot(line_pix, line_data, \
        #           alpha=0.5, color='b', drawstyle="steps", lw=2.)
        # l_ax.plot(line_pix, spectral_line_model(p_opt, line_pix), \
        #           alpha=0.5, color='r', drawstyle="steps", lw=2.)
        # l_fig.savefig(os.path.join(plot_path,
        #               "fit_line_{0}.png".format(int(wvln))))
        # plt.close()

    ax.set_xlim(0,1023)
    ax.set_ylim(0, ylim_max + 1000)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Raw counts")
    fig.savefig(os.path.join(plot_path, "labeled_arc_{0}.pdf".format(night)))

    # TODO: write this out as initial conditions for 2D fit? see marla's code
    zip(fit_pix, arc_lines)

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
