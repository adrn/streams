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
import cosmics
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

        self._data_file_paths = dict()

    def reduce(self, overwrite=False):
        """ Reduce data from this pointing.

        """

        # solve for wavelength at every pixel
        science_wvln = self.wavelength_image
        ccd = self.night.observing_run.ccd

        n_flat = self.night.make_normalized_flat()
        path = self.night.redux_path

        for fn in self.file_paths:
            _data_fn = os.path.splitext(os.path.split(fn)[1])[0]
            data_fn = os.path.join(path, "{0}_2d.fit".format(_data_fn))
            self._data_file_paths[fn] = data_fn

            if os.path.exists(data_fn) and overwrite:
                os.remove(data_fn)

            if os.path.exists(data_fn):
                continue

            # subtract bias
            hdu = fits.open(fn)[0]
            frame_data = hdu.data
            hdr = hdu.header
            frame_data = ccd.zero_correct_frame(frame_data,
                                                self.night.master_zero)

            #TODO: frame.inverse_variance # automatically created?
            variance_image = frame_data*ccd.gain + ccd.read_noise**2
            inv_var = 1./variance_image

            # divide by master normalized flat
            frame_data /= n_flat
            inv_var *= n_flat**2

            # if the exposure was more than 60 seconds, run this cosmic
            #   ray flagger on the CCD data
            if hdr['EXPTIME'] > 60:
                c = cosmics.cosmicsimage(frame_data, gain=ccd.gain,
                                         readnoise=ccd.read_noise,
                                         sigclip=8.0, sigfrac=0.5,
                                         objlim=10.0)
                #c.run(maxiter=6)
                #frame_data = c.cleanarray

            # if exposure time > 60 seconds, do sky subtraction
            if hdr['EXPTIME'] > 60:
                pass

            # extract only the science data region
            science_data = frame_data[ccd.regions["science"]]
            science_inv_var = inv_var[ccd.regions["science"]]

            # write this back out as a 2D image
            new_hdu0 = fits.PrimaryHDU(science_data, header=hdr)
            new_hdu1 = fits.ImageHDU(science_wvln,
                            header=fits.Header([("NAME","wavelength")]))
            new_hdu2 = fits.ImageHDU(science_inv_var,
                            header=fits.Header([("NAME","inverse variance")]))
            hdul = fits.HDUList([new_hdu0,new_hdu1,new_hdu2])
            hdul.writeto(data_fn)

    def combine(self):
        """ Combine multiple exposures of the same object in 2D. """

        for fn in self._data_file_paths.values():
            hdulist = fits.open(fn)

            image_data = hdulist[0].data
            wvln_data = hdulist[1].data
            inv_var = hdulist[2].data

            try:
                all_data += image_data*inv_var**2
                all_inv_var += inv_var**2
            except NameError:
                all_data = image_data*inv_var**2
                all_inv_var = inv_var**2

        all_data /= all_inv_var
        all_inv_var = np.sqrt(all_inv_var)

        ### TEST HACK
        hdu = fits.PrimaryHDU(all_data)
        hdu.writeto("/Users/adrian/Downloads/derp.fits", clobber=True)


    def _read_arcs(self):
        Narcs = len(self.arc_file_paths)

        arcs = None
        for ii,arcfile in enumerate(self.arc_file_paths):
            arc_data = fits.getdata(arcfile,0)
            if arcs is None:
                arcs = np.zeros(arc_data.shape + (Narcs,))

            arcs[...,ii] = arc_data

        self.arcs = arcs

    @property
    def wavelength_image(self):
        """ If this pointing has arcs associated with it, use those to
            get the 2D wavelength solution for the frames. If not, use
            the master arc.
        """
        if len(self.arc_file_paths) > 0:
            return self._solve_2d_wavelength(smooth_length=10)
        else:
            wvln_1d = self.night.observing_run.master_arc.wavelength
            region = self.night.observing_run.ccd.regions["science"]
            ncols = (region[1].stop - region[1].start)
            return np.repeat(wvln_1d[:,np.newaxis], ncols, axis=1)

    def _solve_2d_wavelength(self, smooth_length=10, overwrite=False):
        """ TODO: """

        xx,suffix = os.path.split(self.file_paths[0])
        suffix,xx = os.path.splitext(suffix)

        cache_file = os.path.join(self.night.observing_run.redux_path,
                                  "arc", "arc_{0}.fit".format(suffix))

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
