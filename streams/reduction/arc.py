# coding: utf-8

""" Wavelength calibration using arc lamps """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import logging
import json

# Third-party
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import numpy as np

# Project
from .util import *

# Create logger
logger = logging.getLogger(__name__)

__all__ = ["hand_id_lines", "median_arc", "solve_arc_1d"]

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

def solve_arc_1d(pix, arc_1d, redux_path):
    """ TODO: """
    plot_path = os.path.join(redux_path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    hg_ne_lines = line_list("HgNe")

    hand_id_file = os.path.join(redux_path, "arc", "hand_id.json")
    all_id_file = os.path.join(redux_path, "arc", "all_id.json")

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
        fit_lines.append(wvln)

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

    # TODO: pull plotting out of this function so I can add night label below
    ax.set_xlim(0,1023)
    ax.set_ylim(0, ylim_max + 1000)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Raw counts")
    #ax.set_title("Night: {0}".format(night))
    fig.savefig(os.path.join(plot_path, "labeled_arc.pdf"))

    # write this out as initial conditions for 2D fit
    with open(all_id_file, "w") as f:
        s = dict()
        s["wavelength"] = fit_lines
        s["pixel"] = fit_pix
        f.write(json.dumps(s))

    return fit_pix, fit_lines