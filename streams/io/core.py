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
    #_table = ascii.read(filename)
    _table = np.genfromtxt(filename, names=True)

    if expr is not None:
        idx = numexpr.evaluate(str(expr), _table)
        _table = _table[idx]

    if N is not None and N > 0:
        np.random.shuffle(_table)
        _table = _table[:min(N,len(_table))]

    return _table.copy()

# TODO: should be singleton...
class SimulationData(object):

    def __init__(self, filename):
        """ Represents data from a simulation.

            Parameters
            ----------
            filename : str
                Full path to the particle snapshot.

        """

        if not os.path.exists(filename):
            raise IOError("File {} does not exist!".format(filename))
        self.filename = filename

        # cache
        #self._table = None

        self._gal_colnames = ("x","y","z","vx","vy","vz")
        self._hel_colnames = ("l","b","D","mul","mub","vr")

    def satellite(self, bound_expr, frame="galactocentric",
                  column_names=None, column_units=None):
        """ Return a Particle object for the present-day position of the
            Satellite in the specified reference frame / coordinates.

            Parameters
            ----------
            bound_expr : str
                numexpr expression picking out the still-bound particles.
            frame : str (optional)
                Can be either 'galactocentric' or 'g' or 'heliocentric' or 'h'
            column_names : iterable (optional)
                A list of the column names to read from the table and put in
                Particle.
            column_units : iterable (optional)
                A list of the column units.
        """

        if frame.lower().startswith("g"):
            if column_names is None:
                column_names = self._gal_colnames

        elif frame.lower().startswith("h"):
            if column_names is None:
                column_names = self._hel_colnames

        else:
            raise ValueError("Invalid reference frame.")

        # get the satellite position / velocity from the median of the
        #   bound particle positions/velocities
        tbl = read_table(self.filename, expr=bound_expr)

        cols = []
        for cname,cunit in zip(column_names,column_units):
            col = tbl[cname].copy() * cunit
            cols.append(col)

        return Particle([np.median(c.value) for c in cols],
                        names=column_names,
                        units=[c.unit for c in cols],
                        meta=dict(expr=bound_expr))

    def particles(self, N=None, expr=None, frame="galactocentric",
                  column_names=None, column_units=None, meta_cols=[]):
        """ Return a Particle object with N particles selected from the
            simulation with expression expr in the specified reference
            frame / coordinates.

            Parameters
            ----------
            N : int or None (optional)
                Number of particles to return. None or 0 means 'all'
            expr : str (optional)
                Use numexpr to select out only rows that match criteria.
            frame : str (optional)
                Can be either 'galactocentric' or 'g' or 'heliocentric' or 'h'
            column_names : iterable (optional)
                A list of the column names to read from the table and put in
                Particle.
            column_units : iterable (optional)
                A list of the column units.
            meta_cols : iterable (optional)
                List of columns to add to meta data.
        """

        if frame.lower().startswith("g"):
            if column_names is None:
                column_names = self._gal_colnames

        elif frame.lower().startswith("h"):
            if column_names is None:
                column_names = self._hel_colnames

        else:
            raise ValueError("Invalid reference frame.")

        tbl = read_table(self.filename, expr=expr, N=N)
        #tbl = np.genfromtxt(self.filename, names=True)

        # if no column units specified, try to get them from the table
        # TODO:
        #if column_units is None:
        #    column_units = [tbl[c].unit for c in column_names]

        cols = []
        for cname,cunit in zip(column_names,column_units):
            col = tbl[cname].copy() * cunit
            cols.append(col)

        meta = dict(expr=expr)
        for col in meta_cols:
            meta[col] = tbl[col].copy()

        return Particle(cols, names=column_names,
                        meta=meta)

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
