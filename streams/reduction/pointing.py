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
from .arc import find_line_list, ArcSpectrum

# Create logger
logger = logging.getLogger(__name__)

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

        try:
            n = min([int(float(os.path.split(f)[1][8:12])) for f in self.file_paths])
        except ValueError:
            n = os.path.splitext(os.path.split(f)[1])[0]

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
                avg_col = np.median(col_data, axis=1) # TODO: or mean?

                spec = ArcSpectrum(avg_col)
                spec._hand_id_pix = obs_run.master_arc._hand_id_pix
                spec._hand_id_wvln = obs_run.master_arc._hand_id_wvln
                spec._solve_all_lines(hg_ne, dispersion_fit_order=5)
                spec._fit_solved_lines(order=5)

                residual = spec.pix_to_wavelength(spec._all_line_pix) - \
                           spec._all_line_wvln

                min_num_lines = 20
                failed = False
                while sum(np.fabs(residual) > 0.1) > 0:
                    del_idx = np.argsort(np.fabs(residual) - 0.1)[-1]
                    spec._all_line_pix = np.delete(spec._all_line_pix, del_idx)
                    spec._all_line_wvln = np.delete(spec._all_line_wvln, del_idx)

                    spec._fit_solved_lines(order=5)
                    residual = spec.pix_to_wavelength(spec._all_line_pix) - \
                               spec._all_line_wvln

                    if len(spec._all_line_pix) < min_num_lines:
                        failed = True
                        break

                if failed:
                    plt.clf()
                    plt.subplot(211)
                    plt.plot(spec._all_line_pix, residual)
                    plt.xlim(0,1024)

                    plt.subplot(212)
                    plt.plot(spec._all_line_pix, spec._all_line_wvln, 'ko')
                    pp = np.linspace(0,1024,1000)
                    plt.plot(spec._all_line_pix, spec._all_line_wvln, 'ko')
                    plt.plot(pp, spec.pix_to_wavelength(pp))
                    plt.xlim(0,1024)
                    plt.show()
                    sys.exit(0)

                wavelength_2d[:,i] = spec.pix_to_wavelength(spec.pix)

            hdu = fits.PrimaryHDU(wavelength_2d)
            hdu.writeto(cache_file)

        else:
            wavelength_2d = fits.getdata(cache_file, 0)

        return wavelength_2d
