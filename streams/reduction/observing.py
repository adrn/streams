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
import re
from datetime import datetime

# Third-party
from astropy.time import Time
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
from .arc import find_line_list, ArcSpectrum
from .ccd import *
from .pointing import TelescopePointing

# Create logger
logger = logging.getLogger(__name__)

_line_colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]

class ObservingRun(object):

    def __init__(self, path, ccd, data_path=None, redux_path=None,
                 config_file=None):
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
            config_file : str (optional)
                Path to a queue config file. Default is in
                <data_path>/config.json

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

        # find all nights from the run -- these are subdirectories in
        #   data_path that start with, e.g., m10**13
        self.nights = dict()
        pattr = re.compile("m([0-9]{6})")
        for m_path in glob.glob(os.path.join(self.data_path, "m*")):
            xx,m_date = os.path.split(m_path)
            d = pattr.search(m_date).groups()[0]
            month = int(float(d[:2]))
            day = int(float(d[2:4]))
            yr = int("20" + d[4:])

            utc = Time(datetime(yr,month,day+1), scale="utc")
            self.nights[m_date] = ObservingNight(utc=utc,
                                                 observing_run=self)

        # load the object/queue config
        if config_file is None:
            config_file = os.path.join(self.data_path, "config.json")

        if not os.path.exists(config_file):
            raise IOError("Queue spec/config file '{0}'' does not exist!"\
                          .format(config_file))

        with open(config_file) as f:
            config = json.loads(f.read())

        for night_str,night in self.nights.items():
            for obj in config[night_str]:
                # in json file, i only specify the exposure id. have
                #   to turn this into a filename
                _fn = os.path.join(night.data_path, "{0}.{1:04d}.fit")
                spec_files = [_fn.format(night_str, ii) for ii in obj["spec"]]
                arc_files = [_fn.format(night_str, ii) for ii in obj["arc"]]

                ptg = TelescopePointing(night, spec_files, arc_files)
                night.pointings.append(ptg)

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

        self.master_bias = self.make_master_bias()
        self.master_zero = self.make_master_zero()
        self.master_flat = self.make_master_flat()

        # all pointings of the telescope during this night
        self.pointings = []

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

    def make_master_zero(self, overwrite=False):
        """ Make a master zero frame, store image.

            Parameters
            ----------
            overwrite : bool (optional)
        """
        cache_file = os.path.join(self.redux_path, "master_zero.fits")

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        if not os.path.exists(cache_file):
            all_bias_files = find_all_imagetyp(self.data_path, "ZERO")
            Nbias = len(all_bias_files)

            # get the shape of the 'data' region of the ccd
            ccd = self.observing_run.ccd
            region = ccd.regions["data"]

            all_zero = np.zeros(region.shape + (Nbias,))
            for ii,filename in enumerate(all_bias_files):
                hdu = fits.open(filename)[0]

                if hdu.header["EXPTIME"] > 0:
                    raise ValueError("Bias frame with EXPTIME > 0! ({0})"\
                                     .format(filename))

                all_zero[...,ii] = ccd.overscan_subtract(hdu.data)

            master_zero = np.median(all_zero, axis=-1)
            hdu = fits.PrimaryHDU(master_zero)
            hdu.writeto(cache_file)
        else:
            master_zero = fits.getdata(cache_file)

        return master_zero

    def make_master_flat(self, overwrite=False):
        """ Make a master flat frame, store image.

            Parameters
            ----------
            overwrite : bool (optional)
        """

        zero = self.make_master_zero()
        cache_file = os.path.join(self.redux_path, "master_flat.fits")

        ccd = self.observing_run.ccd

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        all_flats = None
        if not os.path.exists(cache_file):
            all_flat_files = find_all_imagetyp(self.data_path, "FLAT")
            Nflat = len(all_flat_files)

            for ii,filename in enumerate(all_flat_files):
                data = fits.getdata(filename,0)
                corrected = ccd.zero_correct_frame(data, zero)

                if all_flats is None:
                    all_flats = np.zeros(corrected.shape + (Nflat,))

                all_flats[...,ii] = corrected

            master_flat = np.median(all_flats, axis=-1)
            hdu = fits.PrimaryHDU(master_flat)
            hdu.writeto(cache_file)
        else:
            master_flat = fits.getdata(cache_file)

        return master_flat

    def make_normalized_flat(self, overwrite=False):
        """ Make a normalized Flat frame by fitting and dividing out
            the flat lamp spectrum.

            TODO: right now, this just fits a 1D polynomial. could do
                  a 2D fit to the image instead...

            Parameters
            ----------
            overwrite : bool (optional)
        """

        # TODO: I can't get this 2D fit to work...
        # x, y = np.mgrid[:shp[0], :shp[1]]
        # p = models.Polynomial2DModel(degree=13)
        # fit = fitting.LinearLSQFitter(p)
        # fit(x, y, master_flat)
        # smooth_flat = p(x,y)

        cache_file = os.path.join(self.redux_path, "normalized_flat.fits")

        if os.path.exists(cache_file) and overwrite:
            os.remove(cache_file)

        if not os.path.exists(cache_file):
            # collapse the flat to 1D, fit a polynomial
            master_flat_1d = np.median(self.master_flat, axis=1)
            p = models.Polynomial1DModel(degree=15)
            fit = fitting.NonLinearLSQFitter(p)
            pix = np.arange(len(master_flat_1d))
            fit(pix, master_flat_1d)
            smooth_flat = p(pix)

            # normalize the flat
            n_flat = self.master_flat / smooth_flat[:,np.newaxis]
            hdu = fits.PrimaryHDU(n_flat)
            hdu.writeto(cache_file)
        else:
            n_flat = fits.getdata(cache_file)

        return n_flat
