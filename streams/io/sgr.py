# coding: utf-8

""" Classes for accessing simulation data for Sgr-like streams with
    different mass progenitors.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from random import sample

# Third-party
import numpy as np
import numexpr
import astropy.io.ascii as ascii
from astropy.table import Column
import astropy.units as u
from astropy.constants import G
from streamteam.io import SCFReader
from streamteam.units import usys as _usys

# Project
from .. import usys
from ..dynamics import Particle, Orbit
from ..util import streamspath
from ..coordinates.frame import galactocentric
from ..potential.lm10 import LawMajewski2010
from ..inference.util import guess_tail_bit

__all__ = ["SgrSimulation"]

class SgrSimulation(object):

    def __init__(self, path, snapfile):
        """ """

        # potential used for the simulation
        self.potential = LawMajewski2010()

        # some smart pathness
        if not os.path.exists(path):
            _path = os.path.join(streamspath, "data", "simulation", path)
            if os.path.exists(_path):
                path = _path
            else:
                raise IOError("Path '{}' doesn't exist".format(path))
        self.path = path

        self.reader = SCFReader(self.path)
        self.particle_table = self.reader.read_snap(snapfile, units=_usys)

        # get mass column from table
        m = np.array(self.particle_table['m'])*self.particle_table['m'].unit
        self.mass = np.sum(m)
        self.t1 = self.particle_table.meta["timestep"]
        self.t2 = 0.

    def particles(self, n=None, expr=None, tail_bit=False):
        """ Return a Particle object with N particles selected from the
            simulation with expression expr.

            Parameters
            ----------
            n : int or None (optional)
                Number of particles to return. None or 0 means 'all'
            expr : str (optional)
                Use numexpr to select out only rows that match criteria.
            tail_bit : bool (optional)
                Compute tail bit or not.
        """

        if expr is not None:
            expr_idx = numexpr.evaluate(str(expr), self.particle_table)
        else:
            expr_idx = np.ones(len(self.particle_table)).astype(bool)

        table = self.particle_table[expr_idx]
        n_idx = np.array(sample(xrange(len(table)), len(table)))
        if n is not None and n > 0:
            idx = n_idx[:n]
        else:
            idx = n_idx
        table = table[idx]

        # get a list of quantities for each column
        coords = []
        for colname in galactocentric.coord_names:
            coords.append(np.array(table[colname])*table[colname].unit)

        meta = dict(expr=expr)
        meta["tub"] = (np.array(table["tub"])*table["tub"].unit).to(_usys["time"]).value

        # create the particle object
        p = Particle(coords, frame=galactocentric, meta=meta)
        p = p.decompose(usys)

        # guess whether in leading or trailing tail
        if tail_bit:
            p.meta["tail_bit"] = p.tail_bit = guess_tail_bit(p, self.satellite(),
                                                             self.potential,
                                                             self.t1, self.t2, -1.)
        else:
            tail_bit = np.ones(p.nparticles)*np.nan

        return p

    def satellite(self):
        """ Return a Particle object with the present-day position of the
            satellite, computed from the still-bound particles.
        """
        expr_idx = numexpr.evaluate("tub==0", self.particle_table)
        bound = self.particle_table[expr_idx]

        q = []
        for colname in galactocentric.coord_names:
            q.append(np.median(np.array(bound[colname]))*bound[colname].unit)

        meta = dict()
        meta["m0"] = self.mass.to(_usys['mass']).value
        mdot = 3.3*10**(np.floor(np.log10(meta["m0"]))-4)
        meta['mdot'] = mdot

        p = Particle(q, frame=galactocentric, meta=meta)
        return p.decompose(usys)
