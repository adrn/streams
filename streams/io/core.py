# coding: utf-8

""" Code for helping to select stars from the nearby Sgr wraps. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import gc

# Third-party
import numpy as np
import numexpr
import astropy.units as u
from astropy.io import ascii
from astropy.table import vstack, Table, Column
import astropy.coordinates as coord

# Project
from ..coordinates import SgrCoordinates, distance_to_sgr_plane
from ..dynamics import Particle, Orbit

__all__ = ["SimulationData"]

def read_table(filename, expr=None, N=None):
    _table = np.genfromtxt(filename, names=True)

    if expr is not None:
        idx = numexpr.evaluate(str(expr), _table)
        _table = _table[idx]

    if N is not None and N > 0:
        np.random.shuffle(_table)
        _table = _table[:min(N,len(_table))]

    return _table

def add_sgr_coordinates(self):
    # TODO: this is broken
    """ Given a table of catalog data, add columns with Sagittarius
        Lambda and Beta coordinates.

        Parameters
        ----------
        data : astropy.table.Table
            Must contain ra, dec or l, b, and dist columns.
    """

    try:
        pre = coord.Galactic(data['l'],data['b'],unit=(u.deg,u.deg))
    except KeyError:
        pre = coord.ICRS(data['ra'],data['dec'],unit=(u.deg,u.deg))

    sgr = pre.transform_to(SgrCoordinates)
    sgr_plane_dist = np.array(data['dist']) * np.sin(sgr.Beta.radian)

    Lambda = Column(sgr.Lambda.degree, name='Lambda', unit=u.degree)
    Beta = Column(sgr.Beta.degree, name='Beta', unit=u.degree)
    sgr_plane_D = Column(sgr_plane_dist, name='Z_sgr', unit=u.kpc)
    data.add_column(Lambda)
    data.add_column(Beta)
    data.add_column(sgr_plane_D)

    return data
