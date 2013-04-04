# coding: utf-8

""" Classes for accessing various data related to this project. """

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.coordinates as coord
import astropy.units as u

# Project
from ..util import project_root

__all__ = ["LINEAR", "QUEST"]

def _make_npy_file(ascii_file, overwrite=False, ascii_kwargs=dict()):
    """ Make a .npy version of the given ascii file data.

        Parameters
        ----------
        ascii_file : str
            The full path to the ascii file to convert.
        overwrite : bool (optional)
            If True, will overwrite any existing npy files and regenerate
            using the latest ascii data.
        ascii_kwargs : dict
            A dictionary of keyword arguments to be passed to ascii.read().
    """

    filename, ext = os.path.splitext(ascii_file)
    npy_filename = filename + ".npy"

    if os.path.exists(npy_filename) and overwrite:
        os.remove(npy_filename)
    elif os.path.exists(npy_filename) and not overwrite:
        return npy_filename

    data = ascii.read(ascii_file, **ascii_kwargs)
    np.save(npy_filename, np.array(data))
    return npy_filename

class StreamData(object):

    def _set_coordinates(self, ra_name, dec_name):
        """ Given the name of the right ascension column and the name of the
            declination column, set the 'ra' and 'dec' attributes as astropy
            coordinates objects.
        """
        icrs = [coord.ICRSCoordinates(ra,dec,unit=(u.degree,u.degree)) \
                for ra,dec in zip(self.data[ra_name], self.data[dec_name])]

        self.ra = [c.ra for c in icrs]
        self.dec = [c.dec for c in icrs]

class LINEAR(StreamData):

    data = None

    def __init__(self, overwrite=False):
        """ Read in LINEAR RR Lyrae catalog from Branimir. """

        if LINEAR.data == None:
            txt_filename = os.path.join(project_root, "data", "catalog", \
                                        "LINEAR_RRab.txt")

            npy_filename = _make_npy_file(txt_filename, overwrite=overwrite)
            LINEAR.data = np.load(npy_filename)

        self._set_coordinates("ra", "dec")

class QUEST(StreamData):

    data = None

    def __init__(self, overwrite=False):
        """ Read in QUEST RR Lyrae from Vivas et al. 2004 """

        if QUEST.data == None:
            fits_filename = os.path.join(project_root, "data", "catalog", \
                                        "quest_vivas2004_RRL.fits")

            hdulist = fits.open(fits_filename)
            QUEST.data = hdulist[1].data

        self._set_coordinates("RAJ2000", "DEJ2000")

