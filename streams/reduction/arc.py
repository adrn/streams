# coding: utf-8

""" Wavelength calibration using arc lamps """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import json
import logging

# Third-party
from astropy.io import fits
from astropy.modeling import models, fitting
import astropy.units as u
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.optimize import leastsq

# Project
from . import obs_path
from .util import *

_line_colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]

def median_arc(path):
    """ Given a path to a night's data, generate a 1D median
        arc spectrum for a rough wavelength solution.

        Parameters
        ----------
        path : str
            Path to a directory full of FITS files for a night.
    """

    # within this path, find all files with IMAGETYP = comp or
    #   OBJECT = Hg Ne
    comp_files = []
    for filename in glob.glob(os.path.join(path, "*.fit*")):
        hdr = fits.getheader(filename)
        if hdr["IMAGETYP"].lower().strip() != "comp" and \
           hdr["OBJECT"].lower().strip() != "hg ne":
            continue

        comp_files.append(filename)

        # only need 10 files -- more than that is overkill
        if len(comp_files) >= 10:
            break

    if len(comp_files) == 0:
        raise ValueError("Didn't find any COMP or Hg Ne files!"
                         "Are you sure {0} is a valid path?".format(path))

    # get the shape of CCD from the first file
    d = fits.getdata(comp_files[0], 0)

    # select just the central 100 pix
    h,w = d.shape
    L_idx = int(w/2) - 50
    R_idx = int(w/2) + 50
    sub_image = d[:,L_idx:R_idx]

    # make a 3D data structure to hold sub-images for all comps
    all_comps = np.zeros((len(comp_files),) + sub_image.shape)

    # store the already loaded data
    all_comps[0] = sub_image

    for ii,fn in enumerate(comp_files):
        if ii == 0: continue # already done
        all_comps[ii] = fits.getdata(fn, 0)[:,L_idx:R_idx]

    # median over each individual comp
    med_comp = np.median(all_comps, axis=0)

    # median over columns in the subimage
    arc_1d = np.median(med_comp, axis=1)
    pix = np.arange(len(arc_1d))

    return pix, arc_1d

def hand_id(path, Nlines=4, redux_path=os.path.join(obs_path, 'reduction')):
    """ Given a path to a night's data,

    """

    plot_path = os.path.join(redux_path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # TODO: could use a matplotlib color scaler...
    if Nlines > len(_line_colors):
        raise ValueError("Use a max of {0} lines.".format(len(_line_colors)))

    # get pixel array and a medianed, 1D arc spectrum
    pix, arc_1d = median_arc(path)

    # used to split the spectrum into Nlines sections, finds the brightest
    #   line in each section and asks the user to identify it
    sub_div = h // Nlines

    # TODO: When matplotlib has a TextBox widget, or a dropdown, let the
    #   user (me) identify the line on the plot
    fig = plt.figure(figsize=(16,12))
    gs = GridSpec(2, Nlines)

    # top plot is just the full
    top_ax = plt.subplot(gs[0,:])
    top_ax.plot(pix, arc_1d, drawstyle="steps")
    top_ax.set_xlim(0,1023)
    top_ax.set_ylabel("Raw counts")

    line_centers = []
    for ii in range(Nlines):
        color = _line_colors[ii]

        # max pixel index to be center of gaussian fit
        c_idx = np.argmax(plot_comp[ii*sub_div:(ii+1)*sub_div])
        c_idx += sub_div*ii

        try:
            line_data = arc_1d[c_idx-5:c_idx+5]
            line_pix = pix[c_idx-5:c_idx+5]
        except IndexError:
            logger.debug("max value near edge of ccd...weird.")
            continue

        p0 = [min(line_data), np.log10(max(line_data)), 0.5, c_idx]

        # fit a spectral line model to the line data
        p_opt, ier = leastsq(spetral_line_erf, x0=p0, \
                             args=(line_pix, line_data))
        model_line = spectral_line_model(p_opt, line_pix)
        c, log_amplitude, stddev, line_center = p_opt
        line_centers.append(line_center)

        top_ax.plot(line_pix, model_line, \
                     drawstyle="steps", color=color, lw=2.)

        bottom_ax = plt.subplot(gs[1,ii])
        bottom_ax.plot(pix, arc_1d, drawstyle="steps")
        bottom_ax.plot(line_pix, model_line, \
                       drawstyle="steps", color=color, lw=2.)
        bottom_ax.set_xlim(c_idx-10, c_idx+10)
        bottom_ax.set_xlabel("Pixel")
        if ii == 0:
            bottom_ax.set_ylabel("Raw counts")
        else:
            bottom_ax.yaxis.set_visible(False)

    line_id_file = os.path.join(plot_path, "{0}_line_id.png".format(night))
    fig.savefig(line_id_file)

    print("")
    print("Now open: {0}".format(line_id_file))
    print("Identify the colored lines. Default unit is angstrom, but ")
    print("you can input values with units, e.g., 162.124 nanometer.")
    print("")

    line_wavelengths = []
    for ii,color in enumerate(line_colors):
        wvln = raw_input("\t Line {0} ({1} line) wavelength: ".format(ii,
                                                                      color))
        wvln = parse_wavelength(wvln)
        line_wavelengths.append(wvln.to(u.angstrom).value)

    return line_centers, line_wavelengths

########
    # cache the hand identified line list
    with open(os.path.join(night_path, "hand_id_lines.json"), 'w') as f:
        s = dict()
        s['wavelength'] = line_wavelengths
        s['pixel'] = line_centers
        f.write(json.dumps(s))

    return

    p = models.Polynomial1DModel(3)
    fit = fitting.LinearLSQFitter(p)
    fit(line_wavelengths, line_centers)

    print(p)

    # store the plot
    # TODO: maybe save after line ID?
    # fig,ax = plt.subplots(1,1,figsize=(11,8))
    # ax.plot(arc_1d, drawstyle="steps")
    # ax.set_xlim(0,1023)
    # ax.set_xlabel("Pixel")
    # ax.set_ylabel("Raw counts")
    # fig.savefig(os.path.join(plot_path, "{0}_median_comp.png".format(night))

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
