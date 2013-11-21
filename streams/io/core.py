# coding: utf-8

""" Code for helping to select stars from the nearby Sgr wraps. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

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

def _tbl_to_quantity_list(tbl, column_names):
    cols = []
    for cname in column_names:
        col = np.array(tbl[cname]) * tbl[cname].unit
        cols.append(col)

    return cols

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
        self._table = None

        self._gal_colnames = ("x","y","z","vx","vy","vz")
        self._hel_colnames = ("l","b","D","mul","mub","vr")

    def table(self, expr=None):
        if self._table is None:
            self._table = ascii.read(self.filename)

        if expr is not None:
            idx = numexpr.evaluate(str(expr), self._table)
            return self._table[idx]

        return self._table

    def satellite(self, bound_expr, frame="galactocentric",
                  column_names=None):
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
        bound = self.table(bound_expr)
        cols = _tbl_to_quantity_list(bound, column_names)
        return Particle([np.median(c) for c in cols], names=column_names,
                        meta=dict(expr=bound_expr))

    def particles(self, N=None, expr=None, frame="galactocentric",
                  column_names=None):
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
        """

        if frame.lower().startswith("g"):
            if column_names is None:
                column_names = self._gal_colnames

        elif frame.lower().startswith("h"):
            if column_names is None:
                column_names = self._hel_colnames

        else:
            raise ValueError("Invalid reference frame.")

        tbl = self.table(expr)

        if N != None and N > 0:
            idx = np.random.randint(0, len(tbl), N)
            tbl = tbl[idx]

        cols = _tbl_to_quantity_list(tbl, column_names)

        return Particle(cols, names=column_names,
                        meta=dict(expr=expr))

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
