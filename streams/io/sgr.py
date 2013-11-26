# coding: utf-8

""" Classes for accessing simulation data for Sgr-like streams with
    different mass progenitors.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import numexpr
import astropy.io.ascii as ascii
from astropy.table import Column
import astropy.units as u
from astropy.constants import G

# Project
from .. import usys
from ..dynamics import Orbit
from .core import SimulationData, read_table
from ..util import project_root

__all__ = ["SgrSimulation"]

_path = os.path.join(project_root, "data", "simulation", "Sgr")

def _units_from_file(scfpar):
    """ Generate a unit system from an SCFPAR file. """

    with open(scfpar) as f:
        lines = f.readlines()
        length = float(lines[16].split()[0])
        mass = float(lines[17].split()[0])

    GG = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
    X = (GG / length**3 * mass)**-0.5

    length_unit = u.Unit("{0} kpc".format(length))
    mass_unit = u.Unit("{0} M_sun".format(mass))
    time_unit = u.Unit("{:08f} Myr".format(X))

    return dict(length=length_unit,
                mass=mass_unit,
                time=time_unit)

class SgrSimulation(SimulationData):

    def __init__(self, mass, orbit=False):
        """ Data from one of Kathryn's Sgr simulations

            Parameters
            ----------
            mass : str
                e.g., 2.5e8
        """
        filename = os.path.join(_path,mass,"SNAP")
        self.mass = float(mass)
        self._units = _units_from_file(os.path.join(_path,mass,"SCFPAR"))

        super(SgrSimulation, self).__init__(filename=filename)
        self._hel_colnames = ()

        self.t1 = (4.189546E+02 * self._units["time"]).to(u.Myr).value
        self.t2 = 0

        if orbit:
            sgr_orbit = ascii.read(os.path.join(_path,mass,"SCFCEN"))
            for x in "xyz":
                sgr_orbit[x].unit = self._units["length"]

            for x in ("vx","vy","vz"):
                sgr_orbit[x].unit = self._units["length"]/self._units["time"]

            names = ("x","y","z","vx","vy","vz")
            self.satellite_orbit = Orbit(sgr_orbit["t"]*self._units["time"],
              [np.atleast_2d(np.array(sgr_orbit[x])) for x in names],
              names=names,
              units=[sgr_orbit[x].unit for x in names])

    def satellite(self, bound_expr="tub==0", frame="galactocentric",
                  column_names=None):

        col_units = [self._units["length"]]*3 + \
                    [self._units["length"]/self._units["time"]]*3
        s = super(SgrSimulation, self).satellite(bound_expr=bound_expr,
                                                 frame=frame,
                                                 column_names=column_names,
                                                 column_units=col_units)
        s.m = s.meta["m"] = self.mass

        bound = read_table(self.filename, expr=bound_expr)
        vx = np.array((bound["vx"]*col_units[3]).to(u.kpc/u.Myr).value, \
                      copy=True)
        vy = np.array((bound["vy"]*col_units[3]).to(u.kpc/u.Myr).value, \
                      copy=True)
        vz = np.array((bound["vz"]*col_units[3]).to(u.kpc/u.Myr).value, \
                      copy=True)

        del bound
        s.v_disp =s.meta["v_disp"] = np.sqrt(np.var(vx)+np.var(vy)+np.var(vz))

        return s.decompose(usys)

    def particles(self, N=None, expr=None, frame="galactocentric",
                  column_names=None, meta_cols=['tub']):
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
            meta_cols : iterable (optional)
                List of columns to add to meta data.
        """

        col_units = [self._units["length"]]*3 + \
                    [self._units["length"]/self._units["time"]]*3
        p = super(SgrSimulation, self).particles(N=N, expr=expr, frame=frame,
                                                 column_names=column_names,
                                                 column_units=col_units,
                                                 meta_cols=meta_cols)
        p.meta["tub"] = (p.meta["tub"]*self._units["time"])\
                          .decompose(usys).value
        return p.decompose(usys)
