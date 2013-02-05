# coding: utf-8

""" Classes for accessing various data related to this project. """

from __future__ import division, print_function

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

__all__ = ["LM10", "LINEAR", "QUEST", "SgrCen", "SgrSnapshot"]

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

class LM10(StreamData):

    data = None

    def __init__(self, overwrite=False):
        """ Read in simulation data from Law & Majewski 2010.

            Parameters
            ----------
            overwrite : bool (optional)
                If True, will overwrite any existing npy files and regenerate
                using the latest ascii data.
        """

        if LM10.data == None:
            dat_filename = os.path.join(project_root, "data", "simulation", \
                                    "SgrTriax_DYN.dat")

            npy_filename = _make_npy_file(dat_filename, overwrite=overwrite)
            LM10.data = np.load(npy_filename)

        self._set_coordinates("ra", "dec")

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

class SgrCen(StreamData):

    data = None

    def __init__(self, overwrite=False):
        """ Read in Sgr satellite center simulation data from Kathryn """

        # Scale Sgr simulation data to physical units
        ru = 0.63 # kpc
        vu = (41.27781037*u.km/u.s).to(u.kpc/u.Myr).value # kpc/Myr
        tu = 14.9238134129 # Myr

        if SgrCen.data == None:
            txt_filename = os.path.join(project_root, "data", "simulation", \
                                        "SGR_CEN")

            npy_filename = _make_npy_file(txt_filename, overwrite=overwrite, ascii_kwargs=dict(names=["t","dt","x","y","z","vx","vy","vz"]))
            SgrCen.data = np.load(npy_filename)

            SgrCen.data["t"] = SgrCen.data["t"] * tu
            SgrCen.data["dt"] = SgrCen.data["dt"] * tu

            SgrCen.data["x"] = SgrCen.data["x"] * ru
            SgrCen.data["y"] = SgrCen.data["y"] * ru
            SgrCen.data["z"] = SgrCen.data["z"] * ru

            SgrCen.data["vx"] = SgrCen.data["vx"] * vu
            SgrCen.data["vy"] = SgrCen.data["vy"] * vu
            SgrCen.data["vz"] = SgrCen.data["vz"] * vu

class SgrSnapshot(StreamData):

    def __init__(self, overwrite=False, num=0, no_bound=False):
        """ Read in Sgr simulation snapshop for individual particles

            Parameters
            ----------
            num : int
                If 0, load all stars, otherwise randomly sample 'num' particles from the snapshot data.
            no_bound : bool (optional)
                If True, only randomly select particles from the tidal streams -- *not* particles still
                bound to the satellite.
        """

        # Scale Sgr simulation data to physical units
        ru = 0.63 # kpc
        vu = (41.27781037*u.km/u.s).to(u.kpc/u.Myr).value # kpc/Myr
        tu = 14.9238134129 # Myr

        txt_filename = os.path.join(project_root, "data", "simulation", \
                                    "SGR_SNAP")

        npy_filename = _make_npy_file(txt_filename, overwrite=overwrite, ascii_kwargs=dict(names=["m","x","y","z","vx","vy","vz","s1", "s2", "tub"]))
        self.data = np.load(npy_filename)

        if num > 0:

            if no_bound:
                nb_idx = self.data["tub"] > 0.
                data = self.data[nb_idx]
            else:
                data = self.data

            idx = np.random.randint(0, len(data), num)
            self.data = self.data[idx]

        self.data["m"] = self.data["m"]

        self.data["x"] = self.data["x"] * ru
        self.data["y"] = self.data["y"] * ru
        self.data["z"] = self.data["z"] * ru

        self.data["vx"] = self.data["vx"] * vu
        self.data["vy"] = self.data["vy"] * vu
        self.data["vz"] = self.data["vz"] * vu

        self.num = len(self.data)