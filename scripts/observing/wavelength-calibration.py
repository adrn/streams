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
import logging

# Third-party
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.reduction import *

def solve_arc_image(data, L_idx, R_idx, fit_pix, fit_lines):
    """ Given 2D CCD data of an arc lamp and a rough wavelength
        solution from a 1D trace, solve for a 2D wavelength array
        -- that is, wavelength value at each pixel of the image.

        L_idx and R_idx are to specify a sub-section of the data
        to actually solve.

        TODO: do smart things with indices...
    """

    # average over this many columns
    avg_len = 10

    # determine wavelength solution for each column on the CCD
    nrows = len(data)
    pix = np.arange(nrows)
    ncols = R_idx - L_idx

    wavelength_2d = np.zeros((nrows, ncols))
    for i in range(ncols):
        col_fit_pix = []
        col_fit_lines = []

        ii = i + L_idx
        col_data = data[:,ii-avg_len//2:ii+avg_len//2]
        avg_col = np.mean(col_data, axis=1)

        for c_pix,wvln in zip(fit_pix, fit_lines):
            c_idx = int(c_pix)

            line_pix = pix[c_idx-5:c_idx+5]
            line_data = avg_col[c_idx-5:c_idx+5]
            try:
                p_opt = gaussian_fit(line_pix, line_data)
            except ValueError:
                logger.info("Line {0} fit failed.".format(wvln))
                continue

            c, log_amplitude, stddev, line_center = p_opt

            if abs(line_center - c_pix) > 1.:
                logger.info("Line {0} fit failed.".format(wvln))
                continue

            col_fit_pix.append(line_center)
            col_fit_lines.append(wvln)

        col_fit_pix = np.array(col_fit_pix)
        col_fit_lines = np.array(col_fit_lines)

        p = polynomial_fit(col_fit_pix, col_fit_lines, order=5)
        residual = np.fabs(col_fit_lines - p(col_fit_pix))

        # reject where residual >= 0.1 A
        ix = residual < 0.1
        p = polynomial_fit(col_fit_pix[ix], col_fit_lines[ix], order=5)
        wavelength_2d[:,i] = p(pix)

        # fig,axes = plt.subplots(2,1,figsize=(11,8), sharex=True)
        # axes[0].plot(col_fit_pix, col_fit_lines, marker='o', linestyle='none')
        # axes[0].plot(pix, p(pix), linestyle='--', alpha=0.5, color='b')
        # axes[1].plot(col_fit_pix, col_fit_lines-p(col_fit_pix),
        #              marker="o", linestyle="none", color='r', ms=5)
        # axes[1].axhline(0., lw=2.)
        # axes[1].set_xlim(min(pix),max(pix))
        # fig.subplots_adjust(hspace=0.)
        # plt.show()

    return wavelength_2d

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

    run = "2013-10_MDM"
    night = "m102413"

    redux_path = os.path.join(obs_path, run, "reduction")
    data_path = os.path.join(obs_path, run, "data", night)
    pix, arc_1d = median_arc(data_path)

    fit_pix, fit_lines = solve_arc_1d(pix, arc_1d, redux_path)

    # TODO: need to figure out a way to combine consecutive arcs,
    #   average or median them, then use those as data below...
    #   Better yet, write a top-level function given object files,
    #   arcs, flats, biases, etc., have it only solve the arcs for that
    #   object?
    comp_files = find_comp_files(data_path)
    for fn in comp_files:
        # filename to save to in reduction/arc
        xx,just_filename = os.path.split(fn)
        save_file = os.path.join(redux_path, "arc",
                                 "arc_{0}".format(just_filename))

        if os.path.exists(save_file):
            continue

        # get data from comp file
        arc_data = fits.getdata(fn,0)

        # just the central 100 pix
        h,w = arc_data.shape
        L_idx = int(w/2) - 50
        R_idx = int(w/2) + 50

        wavelength_2d = solve_arc_image(arc_data, L_idx, R_idx,
                                        fit_pix, fit_lines)
        hdu = fits.PrimaryHDU(wavelength_2d)
        hdu.writeto(save_file)
