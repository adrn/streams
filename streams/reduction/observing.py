# coding: utf-8

""" Classes for observing / reducing data. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from collections import defaultdict
import glob
import logging
import json

# Third-party
from astropy.io import fits
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Project
from .util import gaussian_fit, polynomial_fit

# Create logger
logger = logging.getLogger(__name__)

def find_all_imagetyp(path, imagetyp):
    """ Find all FITS files in the given path with the IMAGETYP
        header keyword equal to whatever is specified.

        Parameters
        ----------
        path : str
            Path to a bunch of FITS files.
        imagetype : str
            The desired IMAGETYP.
    """

    files = []
    for filename in glob.glob(os.path.join(path, "*.fit*")):
        hdr = fits.getheader(filename,0)
        if hdr["IMAGETYP"] == imagetyp:
            files.append(filename)

    return files

def plot_spectrum(dispersion, flux, fig=None, ax=None, **kwargs):
    """ Plot a spectrum.

        Parameters
        ----------
        dispersion : array_like
        flux : array_like
        fig : matplotlib.Figure
        ax : matplotlib.Axes
    """

    if fig is None and ax is None:
        fig,ax = plt.subplots(1,1,figsize=kwargs.pop("figsize",(11,8)))

    if ax is not None and fig is None:
        fig = ax.figure

    ax.plot(dispersion, flux, drawstyle='steps', **kwargs)
    ax.set_xlim(min(dispersion), max(dispersion))
    ax.set_ylim(0, max(flux) + (max(flux)-min(flux))/20)

    return fig,ax


class CCD(object):

    def __init__(self, shape, gain, read_noise, dispersion_axis=0):
        """ Represents a CCD detector. You should set the following
            additional attributes:

                data_mask : a boolean mask with shape
                    'data[readout_mask].shape' that picks out the region
                    to be used for science;
                overscan_mask : a boolean mask with shape
                    'data[readout_mask].shape' that designates the overscan.

            TODO: gain and read_noise should have units?

            Parameters
            ----------
            shape : tuple
                Number of pixels along each axis.
            gain : numeric
            read_noise : numeric
            dispersion_axis : int (optional)
                Defaults to axis=0. The dispersive axis, e.g., wavelength.
        """

        self.shape = tuple(shape)
        if len(self.shape) != 2:
            raise ValueError("'shape' must be a 2 element iterable.")

        self.gain = float(gain)
        self.read_noise = float(read_noise)
        self.dispersion_axis = int(dispersion_axis)

        # can define named sub-regions of the detector
        self.regions = dict()

    def __getitem__(self, *slices):
        return CCDRegion(self, *slices)

class CCDRegion(list):

    def __init__(self, ccd, *slices):
        """ Represents a region / subset of a CCD detector

            Parameters
            ----------
            ccd : CCD
                The parent CCD object
            slices : tuple
                A tuple of slice objects which define the sub-region by
                slicing along each axis of the CCD.
        """
        self.ccd = CCD
        super(CCDRegion, self).__init__(*slices)


_line_colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]
class ObservingRun(object):

    def __init__(self, path, ccd, data_path=None, redux_path=None):
        """ An object to store global properties of an observing run,
            such as the CCD configuration, a rough wavelength solution,
            various system paths.

            If only 'path' is specified, assumes that all data for that
            observing run live under path/data and reduction products
            will go under path/reduction.

            Parameters
            ----------
            path : str
                Path to the top level of observing run.
            ccd : CCD
                The CCD used to take data.
            data_path : str (optional)
                Path to data for this observing run. Defaults to path/data.
            redux_path : str (optional)
                Path to store reduction products for this observing run.
                Defaults to path/reduction.

        """

        self.path = str(path)
        if not os.path.exists(self.path):
            raise IOError("Path {0} does not exist!".format(self.path))

        if not isinstance(ccd, CCD):
            raise TypeError("'ccd' must be a CCD instance.")
        self.ccd = ccd

        self.data_path = data_path
        if self.data_path is None:
            self.data_path = os.path.join(self.path, "data")

        if not os.path.exists(self.data_path):
            raise IOError("Data path {0} does not exist!"\
                          .format(self.data_path))

        self.redux_path = redux_path
        if self.redux_path is None:
            self.redux_path = os.path.join(self.path, "reduction")

        if not os.path.exists(self.redux_path):
            os.mkdir(self.redux_path)

    def make_master_arc(self, night, narcs=10, overwrite=False):
        """ Make a 'master' 1D arc for this observing run, cache it to
            a JSON file in night.night_path.

            Parameters
            ----------
            night : ObservingNight
            narcs : int (optional)
                Number of arc files to median when processing.
            overwrite : bool (optional)
                Overwrite the cache file or not.
        """
        plt.switch_backend("agg")

        narcs = int(narcs)
        cache_file = os.path.join(self.redux_path, "arc", "master_arc.json")

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        if not os.path.exists(cache_file):
            # first find all COMP files in the data path
            comp_files = find_all_imagetyp(night.night_path, "COMP")
            if len(comp_files) < narcs:
                raise ValueError("Fewer than narcs={0} arcs found in {1}"\
                                 .format(narcs, self.data_path))

            comp_files = comp_files[:narcs]

            # make a 3D data structure to hold data from all comps
            all_comps = np.zeros(self.ccd.shape + (narcs, ))

            for ii,fn in enumerate(comp_files):
                all_comps[...,ii] = fits.getdata(fn, 0)

            # only select out the part of the read-out CCD to use for science
            all_comps = all_comps[self.ccd.regions["data"]]

            # take median over each individual exposure
            median_comp = np.median(all_comps, axis=-1)

            # now take median over the columns in the sub-image
            median_comp_1d = np.median(median_comp, axis=1)
            pix = np.arange(len(median_comp_1d))

            # TODO: pix isn't quite right if select a subsample in the disperson axis...

            with open(cache_file, "w") as f:
                s = dict()
                s["pixel"] = pix.tolist()
                s["arc"] = median_comp_1d.tolist()
                f.write(json.dumps(s))

        with open(cache_file) as f:
            s = json.loads(f.read())

        return np.array(s["pixel"]), np.array(s["arc"])

    def hand_id_lines(self, night, Nlines=4, overwrite=False):
        """ Have the user identify some number of lines by hand.

            Parameters
            ----------
            night : ObservingNight
            Nlines : int
                Number of lines to identify.
            overwrite : bool (optional)
                Overwrite the cache file or not.
        """
        plt.switch_backend("agg")

        # TODO: could use a matplotlib color scaler...
        if Nlines > len(_line_colors):
            raise ValueError("Use a max of {0} lines."\
                             .format(len(_line_colors)))

        # TODO: maybe instead, make_master_arc makes self.pix, self.arc ?
        pix, arc = self.make_master_arc(night)

        cache_file = os.path.join(self.redux_path, "arc", "hand_id.json")

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        if not os.path.exists(cache_file):
            # used to split the spectrum into Nlines sections, finds the brightest
            #   line in each section and asks the user to identify it
            sub_div = len(arc) // Nlines

            # TODO: When matplotlib has a TextBox widget, or a dropdown, let the
            #   user (me) identify the line on the plot
            fig = plt.figure(figsize=(16,12))
            gs = GridSpec(2, Nlines)

            # top plot is just the full
            top_ax = plt.subplot(gs[0,:])
            plot_spectrum(pix, arc, ax=top_ax)
            top_ax.set_ylabel("Raw counts")

            line_centers = []
            for ii in range(Nlines):
                color = _line_colors[ii]

                # max pixel index to be center of gaussian fit
                c_idx = np.argmax(arc[ii*sub_div:(ii+1)*sub_div])
                c_idx += sub_div*ii

                try:
                    line_data = arc[c_idx-5:c_idx+5]
                    line_pix = pix[c_idx-5:c_idx+5]
                except IndexError:
                    logger.debug("max value near edge of ccd...weird.")
                    continue

                g = gaussian_fit(line_pix, line_data)
                line_centers.append(g.mean)

                top_ax.plot(line_pix, g(line_pix), \
                             drawstyle="steps", color=color, lw=2.)

                bottom_ax = plt.subplot(gs[1,ii])
                bottom_ax.plot(pix, arc, drawstyle="steps")
                bottom_ax.plot(line_pix, g(line_pix), \
                               drawstyle="steps", color=color, lw=2.)
                bottom_ax.set_xlim(c_idx-10, c_idx+10)
                bottom_ax.set_xlabel("Pixel")
                if ii == 0:
                    bottom_ax.set_ylabel("Raw counts")
                else:
                    bottom_ax.yaxis.set_visible(False)

            line_id_file = os.path.join(self.redux_path, "plots",
                                        "line_id.pdf")
            fig.savefig(line_id_file)
            plt.close()

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

            with open(cache_file, "w") as f:
                s = dict()
                s["pixel"] = pix.tolist()
                s["wavelength"] = median_comp_1d.tolist()
                f.write(json.dumps(s))

        with open(cache_file) as f:
            s = json.loads(f.read())

        return np.array(s["pixel"]), np.array(s["wavelength"])

    def solve_all_lines(self, night, line_list, overwrite=False):
        """ Now that we have some lines identified by hand, solve for all
            line centers.

            Parameters
            ----------
            night : ObservingNight
            line_list : array_like
                Array of wavelengths of all lines for this arc.
            overwrite : bool (optional)
                Overwrite the cache file or not.
        """

        line_list = np.array(line_list)
        line_list.sort()

        pix, arc = self.make_master_arc(night)
        hand_id_pix, hand_id_wvln = self.hand_id_lines(night)

        cache_file = os.path.join(self.redux_path, "arc", "all_lines.json")

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        if not os.path.exists(cache_file):
            # fit a polynomial to the hand identified lines
            p = polynomial_fit(hand_id_wvln, hand_id_pix, order=3)
            predicted_pix = p(line_list)

            # only get ones within our pixel range
            idx = (predicted_pix>0) & (predicted_pix<1024)
            arc_lines = line_list[idx]
            arc_pix = predicted_pix[idx]

            # fit a gaussian to each line, determine center
            fit_pix = []
            fit_lines = []
            for c_pix,wvln in zip(arc_pix, arc_lines):
                c_idx = int(c_pix)

                line_pix = pix[c_idx-5:c_idx+5]
                line_data = arc[c_idx-5:c_idx+5]
                p = gaussian_fit(line_pix, line_data)

                if abs(p.mean - c_pix) > 1.:
                    logger.info("Line {0} fit failed.".format(wvln))
                    continue

                fit_pix.append(p.mean.value)
                fit_lines.append(wvln)

            with open(cache_file, "w") as f:
                s = dict()
                s["pixel"] = fit_pix
                s["wavelength"] = fit_lines
                f.write(json.dumps(s))

        with open(cache_file) as f:
            s = json.loads(f.read())

        return np.array(s["pixel"]), np.array(s["wavelength"])

class ObservingNight(object):

    def __init__(self, utc, observing_run, night_path=None):
        """ An object to store various properties of a night of an
            observing run, e.g, all bias frames, flats, objects, etc.
            Also, date of run, etc.

            Parameters
            ----------
            utc : astropy.time.Time
                The UT date of the night.
            observing_run : ObservingRun
                The parent ObservingRun object.
            night_path : str (optional)
                Path to data from this night. Defaults to
                <observing_run.data_path>/mMMDDYY where day is *not* the
                UT day of the run.

        """

        self.utc = utc
        self.observing_run = observing_run

        self.night_path = None
        if self.night_path is None:
            # convention is to use civil date at start of night for data,
            #   which is utc day - 1
            day = "{:02d}".format(utc.datetime.day-1)
            month = utc.datetime.strftime("%m")
            year = utc.datetime.strftime("%y")
            night_str = "m{0}{1}{2}".format(month, day, year)
            self.night_path = os.path.join(self.observing_run.data_path,
                                           night_str)

        # create a dict with all unique object names as keys, paths to
        #   files as values
        all_object_files = find_all_imagetyp(self.night_path, "OBJECT")
        if len(all_object_files) == 0:
            raise ValueError("No object files found in '{0}'"\
                             .format(self.night_path))

        object_dict = defaultdict(list)
        for filename in all_object_files:
            hdr = fits.getheader(filename)
            object_dict[hdr["OBJECT"]].append(filename)

        self.object_files = object_dict

