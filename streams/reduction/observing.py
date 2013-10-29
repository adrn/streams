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

# Create logger
logger = logging.getLogger(__name__)

class ObservingRun(object):

    def __init__(self, path, data_path=None, redux_path=None):
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
            data_path : str (optional)
                Path to data for this observing run. Defaults to path/data.
            redux_path : str (optional)
                Path to store reduction products for this observing run.
                Defaults to path/reduction.

        """

        self.path = str(path)
        if not os.path.exists(self.path):
            raise IOError("Path {0} does not exist!".format(self.path))

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

