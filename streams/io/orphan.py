# coding: utf-8

""" Classes for accessing simulation data for an Orphan-like stream """

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
from .core import SimulationData
from ..util import project_root

__all__ = ["OrphanSimulation"]

_path = os.path.join(project_root, "data", "simulation", "Orphan")

X = (G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value / 0.15**3 * 1E7)**-0.5
length_unit = u.Unit("0.15 kpc")
mass_unit = u.Unit("1E7 M_sun")
time_unit = u.Unit("{:08f} Myr".format(X))
orphan_units = dict(length=length_unit,
                    mass=mass_unit,
                    time=time_unit)

class OrphanSimulation(SimulationData):

    def __init__(self, filename=os.path.join(_path,"ORP_SNAP")):
        """ Data from one of Kathryn's Orphan-like simulations

            Parameters
            ----------
            filename : str
        """

        # TODO: self.mass = float(mass) ???
        self._units = orphan_units

        super(OrphanSimulation, self).__init__(filename=filename)
        self._hel_colnames = ()

        self.t1 = (5.189893E+02 * self._units["time"]).to(u.Myr).value
        self.t2 = 0

    def table(self, expr=None):
        if self._table is None:
            tbl = super(OrphanSimulation, self).table()

            for x in "xyz":
                tbl[x].unit = self._units["length"]

            for x in ("vx","vy","vz"):
                tbl[x].unit = self._units["length"]/self._units["time"]

            self._table = tbl

        return super(OrphanSimulation, self).table(expr=expr)

    def satellite(self, bound_expr="tub==0", frame="galactocentric",
                  column_names=None):
        s = super(OrphanSimulation, self).satellite(bound_expr=bound_expr,
                                                 frame=frame,
                                                 column_names=column_names)
        # TODO: s.meta["m"] = self.mass

        bound = self.table(bound_expr)
        s.meta["v_disp"] = np.sqrt(np.std(bound["vx"])**2 + \
                                   np.std(bound["vy"])**2 + \
                                   np.std(bound["vz"])**2)

        return s.decompose(usys)

    def particles(self, *args, **kwargs):
        p = super(OrphanSimulation, self).particles(*args, **kwargs)
        return p.decompose(usys)
