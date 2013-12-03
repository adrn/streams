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
from ..dynamics import Particle, Orbit
from .core import read_table
from ..util import project_root
from ..coordinates.frame import galactocentric

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

class SgrSimulation(object):

    def __init__(self, mass):

        self.particle_filename = os.path.join(_path,mass,"SNAP")
        self.particle_columns = ("x","y","z","vx","vy","vz")

        self._units = _units_from_file(os.path.join(_path,mass,"SCFPAR"))
        self.particle_units = [self._units["length"]]*3 + \
                              [self._units["length"]/self._units["time"]]*3

        self.mass = mass
        self.t1 = (4.189546E+02 * self._units["time"]).decompose(usys).value
        self.t2 = 0

    def raw_particle_table(self, N=None, expr=None):
        tbl = read_table(self.particle_filename, N=N, expr=expr)
        return tbl

    def particles(self, N=None, expr=None, meta_cols=[]):
        """ Return a Particle object with N particles selected from the
            simulation with expression expr.

            Parameters
            ----------
            N : int or None (optional)
                Number of particles to return. None or 0 means 'all'
            expr : str (optional)
                Use numexpr to select out only rows that match criteria.
            meta_cols : iterable (optional)
                List of columns to add to meta data.
        """
        tbl = self.raw_particle_table(N=N, expr=expr)

        q = []
        for colname,unit in zip(self.particle_columns, self.particle_units):
            q.append(np.array(tbl[colname])*unit)

        meta = dict(expr=expr)
        meta["tub"] = (np.array(tbl["tub"])*self._units["time"]).decompose(usys).value
        for col in meta_cols:
            meta[col] = np.array(tbl[col])

        p = Particle(q, frame=galactocentric, meta=meta)
        return p.decompose(usys)

    def satellite(self):
        """ Return a Particle object with the present-day position of the
            satellite, computed from the still-bound particles.
        """
        expr = "tub==0"
        tbl = self.raw_particle_table(expr=expr)

        q = []
        for colname in self.particle_columns:
            q.append(tbl[colname].tolist())

        q = np.array(q)

        meta = dict(expr=expr)
        v_disp = np.sqrt(np.sum(np.var(q[3:],axis=1)))
        meta["v_disp"] = (v_disp*self.particle_units[-1]).decompose(usys).value
        meta["m"] = float(self.mass)

        q = np.median(q, axis=1)
        p = Particle(q, frame=galactocentric,
                     units=self.particle_units,
                     meta=meta)
        return p.decompose(usys)

    @property
    def satellite_orbit(self):
        """ TODO: """

        orbit_table = np.genfromtxt(os.path.join(_path,self.mass,"SCFCEN"), names=True)

        q = []
        for colname,unit in zip(self.particle_columns, self.particle_units):
            q.append(np.array(orbit_table[colname])[:,np.newaxis]*unit)

        return Orbit(orbit_table["t"]*self._units["time"], q,
                     frame=galactocentric)