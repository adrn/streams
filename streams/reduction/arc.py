# coding: utf-8

""" Wavelength calibration using arc lamps """

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
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import numpy as np

# Project
from .util import *

__all__ = ["hand_id_lines", "median_arc"]

_line_colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]

def median_arc(data_path):
    """ Given a path to a night's data, generate a 1D median
        arc spectrum for a rough wavelength solution.

        Parameters
        ----------
        data_path : str
            Path to a directory full of FITS files for a night.
    """

    # within this path, find all files with IMAGETYP = comp or
    #   OBJECT = Hg Ne
    comp_files = []
    for filename in glob.glob(os.path.join(data_path, "*.fit*")):
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
                        "Are you sure {0} is a valid path?".format(data_path))

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

def hand_id_lines(pix, arc_1d, plot_path, Nlines=4):
    """ Given a path to a night's data,

    """

    # TODO: could use a matplotlib color scaler...
    if Nlines > len(_line_colors):
        raise ValueError("Use a max of {0} lines.".format(len(_line_colors)))

    # used to split the spectrum into Nlines sections, finds the brightest
    #   line in each section and asks the user to identify it
    sub_div = len(arc_1d) // Nlines

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
        c_idx = np.argmax(arc_1d[ii*sub_div:(ii+1)*sub_div])
        c_idx += sub_div*ii

        try:
            line_data = arc_1d[c_idx-5:c_idx+5]
            line_pix = pix[c_idx-5:c_idx+5]
        except IndexError:
            logger.debug("max value near edge of ccd...weird.")
            continue

        p_opt = gaussian_fit(line_pix, line_data)
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

    line_id_file = os.path.join(plot_path, "line_id.png")
    fig.savefig(line_id_file)

    print("")
    print("Now open: {0}".format(line_id_file))
    print("Identify the colored lines. Default unit is angstrom, but ")
    print("you can input values with units, e.g., 162.124 nanometer.")
    print("")

    line_wavelengths = []
    for ii,color in enumerate(_line_colors[:Nlines]):
        wvln = raw_input("\t Line {0} ({1} line) wavelength: ".format(ii,
                                                                      color))
        wvln = parse_wavelength(wvln)
        line_wavelengths.append(wvln.to(u.angstrom).value)

    return line_centers, line_wavelengths
