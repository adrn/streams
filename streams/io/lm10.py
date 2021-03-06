# coding: utf-8

""" Classes for accessing simulation data related to Sagittarius. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import numexpr
import astropy.units as u
from astropy.constants import G

# Project
from .core import read_table
from ..util import project_root
from ..dynamics import Particle
from ..coordinates.frame import galactocentric

__all__ = ["LM10Simulation"]

_lm10_path = os.path.join(project_root, "data", "simulation", "LM10")

class LM10Simulation(object):

    def __init__(self):
        self.particle_filename = os.path.join(_lm10_path,"SgrTriax_DYN.dat")
        self.particle_columns = ("xgc","ygc","zgc","u","v","w")
        self.particle_units = (u.kpc, u.kpc, u.kpc,
                               u.km/u.s, u.km/u.s, u.km/u.s)

        self.t1 = 0.
        self.t2 = -8000.

    def raw_particle_table(self, N=None, expr=None):
        tbl = read_table(self.particle_filename, N=N, expr=expr)
        tbl["xgc"] = -tbl["xgc"]
        tbl["u"] = -tbl["u"]
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
        for col in meta_cols:
            meta[col] = np.array(tbl[col])

        p = Particle(q, frame=galactocentric, meta=meta)
        return p

    def satellite(self):
        """ Return a Particle object with the present-day position of the
            satellite, computed from the still-bound particles.
        """
        expr = "Pcol==-1"
        tbl = self.raw_particle_table(expr=expr)

        q = []
        for colname in self.particle_columns:
            q.append(tbl[colname].tolist())

        q = np.array(q)
        q = np.median(q, axis=1)

        p = Particle(q, frame=galactocentric, units=self.particle_units,
                     meta=dict(expr=expr))
        return p