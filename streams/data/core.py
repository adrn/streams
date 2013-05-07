# coding: utf-8

""" Classes for accessing various data related to this project. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy.table import Table, Column
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.coordinates as coord
import astropy.units as u

# Project
from ..util import project_root

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
