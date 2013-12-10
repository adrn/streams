# coding: utf-8

""" Code for helping to select stars from the nearby Sgr wraps. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import gc

# Third-party
import h5py
import numpy as np
import numexpr
import astropy.units as u
from astropy.io import ascii
from astropy.table import vstack, Table, Column
import astropy.coordinates as coord

# Project
from ..coordinates import SgrCoordinates, distance_to_sgr_plane
from ..dynamics import Particle, Orbit

__all__ = ["read_table", "read_hdf5"]

def read_table(filename, expr=None, N=None):
    _table = np.genfromtxt(filename, names=True)

    if expr is not None:
        idx = numexpr.evaluate(str(expr), _table)
        _table = _table[idx]

    if N is not None and N > 0:
        np.random.shuffle(_table)
        _table = _table[:min(N,len(_table))]

    return _table

def read_hdf5(h5file):
    """ Read particles and satellite from a given HDF5 file. """

    with h5py.File(h5file, "r") as f:
        try:
            ptcl = f["particles"]
            satl = f["satellite"]
        except KeyError:
            raise ValueError("Invalid HDF5 file. Missing 'particles' or "
                             "'satellite' group.")

        print(ptcl)
        print(satl)