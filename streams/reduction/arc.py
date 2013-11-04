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

_linelist_path = ("/Users/adrian/Documents/GraduateSchool/Observing/"
                  "MDM 2.4m/line lists")

def find_line_list(name):
    """ Read in a list of wavelengths for a given arc lamp.

        Parameters
        ----------
        name : str
            Name of the arc lamp, e.g., Hg Ne

    """

    try:
        fn = os.path.join(_linelist_path, name.replace(" ", "_")+".txt")
        lines = np.loadtxt(fn)
    except:
        raise ValueError("No list for {0}".format(name))

    return lines

class Spectrum(object):

    def __init__(self, dispersion, intensity):
        """ TODO: """

        self.dispersion = dispersion
        self.intensity = intensity

    def plot(self, fig=None, ax=None, **kwargs):
        """ Plot the spectrum.

            Parameters
            ----------
            fig : matplotlib.Figure
            ax : matplotlib.Axes
        """

        if fig is None and ax is None:
            fig,ax = plt.subplots(1,1,figsize=kwargs.pop("figsize",(11,8)))

        if ax is not None and fig is None:
            fig = ax.figure

        if hasattr(self.dispersion, "unit"):
            disp_unit = self.dispersion.unit
            if disp_unit.physical_type == "length":
                xlbl = "Wavelength"
            else:
                xlbl = disp_unit.physical_type.capitalize()

            x = self.dispersion.value

        else:
            xlbl = "Pixels"
            x = self.dispersion

        if hasattr(self.intensity, "unit"):
            disp_unit = self.dispersion.unit
            ylbl = disp_unit.physical_type.capitalize()
            y = self.intensity.value

        else:
            ylbl = "Counts"
            y = self.intensity

        ax.plot(x, y, drawstyle='steps', **kwargs)
        h = (max(y)-min(y))/10

        ax.set_xlim(min(x), max(x))
        ax.set_ylim(0, max(y) + 3*h)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)

        return fig,ax


class ArcSpectrum(Spectrum):

    @classmethod
    def from_json(cls, filename):
        """ Create an ArcSpectrum from a JSON file. The JSON file
            should have at minimum a 'counts' field, but could also
            have 'wavelength'.
        """

        with open(filename) as f:
            s = json.loads(f.read())

        if not s.has_key("counts"):
            raise ValueError("Invalid JSON file. Must contain counts field.")

        o = cls(s["counts"], wavelength=s.get("wavelength", None),
                cache=filename)
        o._hand_id_pix = s.get("_hand_id_pix", None)
        o._hand_id_wvln = s.get("_hand_id_wvln", None)
        o._all_line_pix = s.get("_all_line_pix", None)
        o._all_line_wvln = s.get("_all_line_wvln", None)
        return o

    def to_json(self):
        s = dict()
        if self.wavelength is not None:
            s["wavelength"] = self.wavelength.tolist()
            s["_hand_id_pix"] = self._hand_id_pix.tolist()
            s["_hand_id_wvln"] = self._hand_id_wvln.tolist()
            s["_all_line_pix"] = self._all_line_pix.tolist()
            s["_all_line_wvln"] = self._all_line_wvln.tolist()
        s["counts"] = self.counts.tolist()
        s["pix"] = self.pix.tolist()

        return json.dumps(s)

    def __init__(self, counts, wavelength=None, cache=None):
        """ Represents the spectrum from an arc lamp.

            Parameters
            ----------
            counts : array_like
                Raw counts from the detector.
            wavelength : array_like (optional)
                If the wavelength grid is already solved.
            cache : str
                File to cache things to.
        """

        self.counts = np.array(counts)

        self.wavelength = wavelength
        if self.wavelength is not None:
            self.wavelength = np.array(self.wavelength)
            assert self.wavelength.shape == self.counts.shape

        self.pix = np.arange(len(self.counts))
        self.cache = cache

    def clear_cache(self):
        os.remove(self.cache)

    def __len__(self):
        return len(self.pix)

    def plot(self, line_ids=False, fig=None, ax=None, **kwargs):
        """ Plot the spectrum.

            Parameters
            ----------
            line_ids : bool
                Put markers over each line with the wavelength.
            fig : matplotlib.Figure
            ax : matplotlib.Axes
        """

        if fig is None and ax is None:
            fig,ax = plt.subplots(1,1,figsize=kwargs.pop("figsize",(11,8)))

        if ax is not None and fig is None:
            fig = ax.figure

        if self.wavelength is not None:
            x = self.wavelength
            ax.set_xlabel("Wavelength")
        else:
            x = self.pix
            ax.set_xlabel("Pixels")

        ax.plot(x, self.counts, drawstyle='steps', **kwargs)
        h = (max(self.counts)-min(self.counts))/10

        if line_ids:
            if self.wavelength is None:
                raise ValueError("Lines have not been identified yet!")

            ys = []
            for px,wvln in zip(self._all_line_pix, self._all_line_wvln):
                ys.append(max(self.counts[int(px) + np.arange(-1,2)]))
                ax.text(wvln-5., ys[-1]+6200.,
                        s="{:.3f}".format(wvln),
                        rotation=90, fontsize=10)

            ys = np.array(ys)
            ax.vlines(self._all_line_wvln, ys+h/3, ys+h, color='#3182BD')

        ax.set_xlim(min(x), max(x))
        ax.set_ylim(0,
                    max(self.counts) + 3*h)
        ax.set_ylabel("Raw counts")

        return fig,ax

    def _hand_id_lines(self, obs_run, Nlines=4):
        """ Have the user identify some number of lines by hand.

            Parameters
            ----------
            obs_run : ObservingRun
                The observing run object.
            Nlines : int
                Number of lines to identify.
        """

        # TODO: could use a matplotlib color scaler...
        if Nlines > len(_line_colors):
            raise ValueError("Use a max of {0} lines."\
                             .format(len(_line_colors)))

        # used to split the spectrum into Nlines sections, finds the brightest
        #   line in each section and asks the user to identify it
        sub_div = len(self) // Nlines

        _bkend = plt.get_backend()
        plt.switch_backend('agg')
        # TODO: When matplotlib has a TextBox widget, or a dropdown, let the
        #   user (me) identify the line on the plot
        fig = plt.figure(figsize=(16,12))
        gs = GridSpec(2, Nlines)

        # top plot is just the full
        top_ax = plt.subplot(gs[0,:])
        self.plot(ax=top_ax)
        top_ax.set_ylabel("Raw counts")

        line_centers = []
        for ii in range(Nlines):
            color = _line_colors[ii]

            # max pixel index to be center of gaussian fit
            c_idx = np.argmax(self.counts[ii*sub_div:(ii+1)*sub_div])
            c_idx += sub_div*ii

            try:
                line_data = self.counts[c_idx-5:c_idx+5]
                line_pix = self.pix[c_idx-5:c_idx+5]
            except IndexError:
                logger.debug("max value near edge of ccd...weird.")
                continue

            g = gaussian_fit(line_pix, line_data)
            line_centers.append(g.mean)

            top_ax.plot(line_pix, g(line_pix), \
                         drawstyle="steps", color=color, lw=2.)

            bottom_ax = plt.subplot(gs[1,ii])
            bottom_ax.plot(self.pix, self.counts, drawstyle="steps")
            bottom_ax.plot(line_pix, g(line_pix), \
                           drawstyle="steps", color=color, lw=2.)
            bottom_ax.set_xlim(c_idx-10, c_idx+10)
            bottom_ax.set_xlabel("Pixel")
            if ii == 0:
                bottom_ax.set_ylabel("Raw counts")
            else:
                bottom_ax.yaxis.set_visible(False)

        line_id_file = os.path.join(obs_run.redux_path, "plots",
                                    "line_id.pdf")
        fig.savefig(line_id_file)
        plt.clf()

        print("")
        print("Now open: {0}".format(line_id_file))
        print("Identify the colored lines. Default unit is angstrom, but ")
        print("you can input values with units, e.g., 162.124 nanometer.")
        print("")

        line_wavelengths = []
        for ii,color in enumerate(_line_colors[:Nlines]):
            wvln = raw_input("\t Line {0} ({1} line) wavelength: "\
                             .format(ii, color))
            wvln = parse_wavelength(wvln)
            line_wavelengths.append(wvln.to(u.angstrom).value)

        plt.switch_backend(_bkend)

        self._hand_id_pix = np.array(line_centers)
        self._hand_id_wvln = np.array(line_wavelengths)

    def _solve_all_lines(self, line_list, dispersion_fit_order=3):
        """ Now that we have some lines identified by hand, solve for all
            line centers.

            Parameters
            ----------
            line_list : array_like
                Array of wavelengths of all lines for this arc.
            dispersion_fit_order : int
                Order of the polynomial fit to the hand identified lines.
        """

        line_list = np.array(line_list)
        line_list.sort()

        # fit a polynomial to the hand identified lines
        p = polynomial_fit(self._hand_id_wvln, self._hand_id_pix,
                           order=dispersion_fit_order)
        predicted_pix = p(line_list)

        # only get ones within our pixel range
        idx = (predicted_pix>0) & (predicted_pix<1024)
        arc_lines = line_list[idx]
        arc_pix = predicted_pix[idx]

        # fit a gaussian to each line, determine center
        fit_pix = []
        fit_wvln = []
        for c_pix,wvln in zip(arc_pix, arc_lines):
            w = 4
            l = int(np.floor(round(c_pix) - w))
            r = int(np.ceil(round(c_pix) + w))

            line_pix = self.pix[l:r+1]
            line_data = self.counts[l:r+1]
            p = gaussian_fit(line_pix, line_data)

            if abs(p.mean - c_pix) > 1.4:
                logger.info("Line {0} fit failed.".format(wvln))
                continue

            fit_pix.append(p.mean.value)
            fit_wvln.append(wvln)

        self._all_line_pix = np.array(fit_pix)
        self._all_line_wvln = np.array(fit_wvln)

    def _fit_solved_lines(self, order=5):
        """ Fit pixel center vs. wavelength for all solved lines.

            Parameters
            ----------
            order : int (optional)
                Order of the polynomial fit to the lines.
        """
        p = polynomial_fit(self._all_line_pix, self._all_line_wvln,
                           order=order)
        self.pix_to_wavelength = p

    def solve_wavelength(self, obs_run, line_list):
        """ Find the wavelength solution.

            Parameters
            ----------
            obs_run : ObservingRun
            line_list : array_like
                List of wavelengths for lines in this arc.
        """

        self._hand_id_lines(obs_run)
        self._solve_all_lines(line_list)
        self._fit_solved_lines()

        self.wavelength = self.pix_to_wavelength(self.pix)

        with open(self.cache, "w") as f:
            f.write(self.to_json())