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
from astropy.io.misc import fnpickle, fnunpickle
from astropy.modeling import models, fitting
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

# Project
from .util import *

# Create logger
logger = logging.getLogger(__name__)

_line_colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]

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

def plot_wavelength_solution(obs_run, night):
    """ TODO: """

    # label the lines, IRAF style
    fig,ax = plt.subplots(1,1,figsize=(11,8))
    ax.plot(pix, arc_1d, drawstyle="steps", c='k')

    all_pix, all_wvln = obs_run.solve_all_lines(night)
    for pix,wvln in zip(all_pix, all_wvln):
        y = max(spectral_line_model(p_opt, line_pix)) + 3000.
        ylim_max = max(ylim_max, y)
        ax.text(line_center-6, y, "{0:.3f}".format(wvln),
                rotation=90, fontsize=10)

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

    def bias_correct_frame(self, frame_data, bias):
        """ Bias subtract and overscan subtract """

        # subtract bias frame
        #data = frame_data - bias
        data = frame_data
        overscan = data[self.regions["overscan"]]
        overscan_col = np.median(overscan, axis=1)

        data -= overscan_col[:,np.newaxis]

        return data[self.regions["data"]]

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

class Spectrum(object):
    pass

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
        return o

    def to_json(self):
        s = dict()
        if self.wavelength is not None:
            s["wavelength"] = self.wavelength.tolist()
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
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(0,
                    max(self.counts) + (max(self.counts)-min(self.counts))/20)
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
            c_idx = int(c_pix)

            line_pix = self.pix[c_idx-5:c_idx+5]
            line_data = self.counts[c_idx-5:c_idx+5]
            p = gaussian_fit(line_pix, line_data)

            if abs(p.mean - c_pix) > 1.:
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
        p = polynomial_fit(self._all_line_pix, self._all_line_wvln)
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

        self.master_arc = None

    def make_master_arc(self, night, narcs=10, overwrite=False):
        """ Make a 'master' 1D arc for this observing run, cache it to
            a JSON file in night.data_path.

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
        cache_file = os.path.join(self.redux_path, "arc", \
                                  "master_arc.json")

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        if not os.path.exists(cache_file):
            # first find all COMP files in the data path
            comp_files = find_all_imagetyp(night.data_path, "COMP")
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
                s["counts"] = median_comp_1d.tolist()
                f.write(json.dumps(s))

        self.master_arc = ArcSpectrum.from_json(cache_file)

class ObservingNight(object):

    def __init__(self, utc, observing_run, data_path=None, redux_path=None):
        """ An object to store various properties of a night of an
            observing run, e.g, all bias frames, flats, objects, etc.
            Also, date of run, etc.

            Parameters
            ----------
            utc : astropy.time.Time
                The UT date of the night.
            observing_run : ObservingRun
                The parent ObservingRun object.
            data_path : str (optional)
                Path to data from this night. Defaults to
                <observing_run.data_path>/mMMDDYY where day is *not* the
                UT day of the run.
            redux_path : str (optional)

        """

        self.utc = utc
        self.observing_run = observing_run

        # convention is to use civil date at start of night for data,
        #   which is utc day - 1
        day = "{:02d}".format(utc.datetime.day-1)
        month = utc.datetime.strftime("%m")
        year = utc.datetime.strftime("%y")
        self._night_str = "m{0}{1}{2}".format(month, day, year)

        if data_path is None:
            data_path = os.path.join(self.observing_run.data_path,
                                     str(self))
        self.data_path = data_path

        if redux_path is None:
            redux_path = os.path.join(self.observing_run.redux_path,
                                      str(self))
        self.redux_path = redux_path

        if not os.path.exists(data_path):
            raise ValueError("Path to data ({0}) doesn't exist!"\
                             .format(self.data_path))

        if not os.path.exists(redux_path):
            os.mkdir(redux_path)

        # create a dict with all unique object names as keys, paths to
        #   files as values
        all_object_files = find_all_imagetyp(self.data_path, "OBJECT")
        if len(all_object_files) == 0:
            raise ValueError("No object files found in '{0}'"\
                             .format(self.data_path))

        object_dict = defaultdict(list)
        for filename in all_object_files:
            hdr = fits.getheader(filename)
            object_dict[hdr["OBJECT"]].append(filename)

        self.object_files = object_dict

    def __str__(self):
        return self._night_str

    def make_master_bias(self, overwrite=False):
        """ Make a master bias frame, store image.

            Parameters
            ----------
            overwrite : bool (optional)
        """

        cache_file = os.path.join(self.redux_path, "master_bias.fits")

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        if not os.path.exists(cache_file):
            all_bias_files = find_all_imagetyp(self.data_path, "ZERO")
            Nbias = len(all_bias_files)

            all_bias = np.zeros(self.observing_run.ccd.shape + (Nbias,))
            for ii,filename in enumerate(all_bias_files):
                hdu = fits.open(filename)[0]

                if hdu.header["EXPTIME"] > 0:
                    raise ValueError("Bias frame with EXPTIME > 0! ({0})"\
                                     .format(filename))

                all_bias[...,ii] = hdu.data

            master_bias = np.median(all_bias, axis=-1)
            hdu = fits.PrimaryHDU(master_bias)
            hdu.writeto(cache_file)
        else:
            master_bias = fits.getdata(cache_file)

        return master_bias

    def make_master_flat(self, overwrite=False):
        """ Make a master flat frame, store image.

            Parameters
            ----------
            overwrite : bool (optional)
        """

        bias = self.make_master_bias()
        cache_file = os.path.join(self.redux_path, "master_flat.fits")

        ccd = self.observing_run.ccd

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        # TODO: make sure flat combination is correct! this is probably where the error is...in making the master flat
        all_flats = None
        if not os.path.exists(cache_file):
            all_flat_files = find_all_imagetyp(self.data_path, "FLAT")
            Nflat = len(all_flat_files)

            for ii,filename in enumerate(all_flat_files):
                data = fits.getdata(filename,0)
                corrected = ccd.bias_correct_frame(data, bias)

                if all_flats is None:
                    all_flats = np.zeros(corrected.shape + (Nflat,))

                all_flats[...,ii] = corrected

            master_flat = np.median(all_flats, axis=-1)
            hdu = fits.PrimaryHDU(master_flat)
            hdu.writeto(cache_file)
        else:
            master_flat = fits.getdata(cache_file)

        return master_flat

class CCDFrame(object):
    pass

class TelescopePointing(object):

    def __init__(self, night, files, arc_files=[]):
        """ Represents a telescope pointing to an observed astronomical
            object. This is important because we may observe the same object
            multiple times in a night, but at different pointings. Each
            pointing has associated arc/comp lamps so we have to distinguish.

            Parameters
            ----------
            night : ObservingNight
            files : list
                List of data files for the object at this pointing.
            arc_files : list
                List of arc lamp files for this pointing.
        """
        self.night = night

        self.file_paths = []
        for fn in files:
            path,filename = os.path.split(fn)
            if len(path.strip()) == 0:
                fn = os.path.join(self.night.data_path, fn)

            if not os.path.exists(fn):
                raise IOError("File {0} does not exist!".format(fn))

            self.file_paths.append(fn)

        self.arc_file_paths = []
        for fn in arc_files:
            path,filename = os.path.split(fn)
            if len(path.strip()) == 0:
                fn = os.path.join(self.night.data_path, fn)

            if not os.path.exists(fn):
                raise IOError("File {0} does not exist!".format(fn))

            self.arc_file_paths.append(fn)

        self.object_name = fits.getheader(self.file_paths[0])["OBJECT"]
        #self._read_arcs()
        #self._read_data()

    def _read_arcs(self):
        Narcs = len(self.arc_file_paths)

        arcs = None
        for ii,arcfile in enumerate(self.arc_file_paths):
            arc_data = fits.getdata(arcfile,0)
            if arcs is None:
                arcs = np.zeros(arc_data.shape + (Narcs,))

            arcs[...,ii] = arc_data

        self.arcs = arcs

    def _read_data(self):
        Ndata = len(self.file_paths)

        all_objects = None
        for ii,fn in enumerate(self.file_paths):
            data = fits.getdata(fn,0)
            if all_objects is None:
                all_objects = np.zeros(data.shape + (Ndata,))

            all_objects[...,ii] = data

        self.all_objects = all_objects

    def solve_2d_wavelength(self, smooth_length=10, overwrite=False):
        """ TODO: """

        n = min([int(float(os.path.split(f)[1][8:12])) for f in self.file_paths])
        cache_file = os.path.join(self.night.redux_path,
                                  "{0}_{1}.fits".format(self.object_name, n))

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        if not os.path.exists(cache_file):
            hg_ne = find_line_list("Hg Ne")
            obs_run = self.night.observing_run

            self._read_arcs()
            all_arc_data = np.median(self.arcs, axis=-1)
            sci_arc_data = all_arc_data[obs_run.ccd.regions["science"]]
            col_idx = (obs_run.ccd.regions["science"][1].start,
                       obs_run.ccd.regions["science"][1].stop)

            # determine wavelength solution for each column on the CCD
            pix = np.arange(all_arc_data.shape[0])

            wavelength_2d = np.zeros_like(sci_arc_data)
            for i in range(sci_arc_data.shape[1]):
                ii = col_idx[0] + i
                col_fit_pix = []
                col_fit_lines = []

                w = smooth_length//2
                col_data = all_arc_data[:,ii-w:ii+w]
                avg_col = np.mean(col_data, axis=1) # TODO: or median?

                spec = ArcSpectrum(avg_col)
                spec._hand_id_pix = obs_run.master_arc._hand_id_pix
                spec._hand_id_wvln = obs_run.master_arc._hand_id_wvln
                spec._solve_all_lines(hg_ne, dispersion_fit_order=3)
                spec._fit_solved_lines(order=5)

                residual = spec.pix_to_wavelength(spec._all_line_pix) - \
                           spec._all_line_wvln
                idx = np.fabs(residual) < 0.1
                spec._all_line_pix = spec._all_line_pix[idx]
                spec._all_line_wvln = spec._all_line_wvln[idx]

                assert len(spec._all_line_pix) > 20
                spec._fit_solved_lines(order=5)

                wavelength_2d[:,i] = spec.pix_to_wavelength(spec.pix)

            hdu = fits.PrimaryHDU(wavelength_2d)
            hdu.writeto(cache_file)

        else:
            wavelength_2d = fits.getdata(cache_file, 0)

        return wavelength_2d
