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

def solve_arc_image(data):
    """ Given 2D CCD data of an arc lamp and a rough wavelength
        solution from a 1D trace, solve for a 2D wavelength array
        -- that is, wavelength value at each pixel of the image.
    """

    # determine wavelength solution for each column on the CCD
    nrows, ncols = data.shape
    pix = np.arange(nrows)

    print(pix)

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
