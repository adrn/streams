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
from .core import SimulationData
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

    def __init__(self, mass):
        """ Data from one of Kathryn's Sgr simulations

            Parameters
            ----------
            mass : str
                e.g., 2.5e8
        """
        filename = os.path.join(_path,mass,"SNAP"))
        self.mass = float(mass)
        self._units = _units_from_file(os.path.join(_path,mass,"SCFPAR")))

        super(SgrSimulation, self).__init__(filename=filename)
        self._hel_colnames = ()

        self.t1 = (4.189546E+02 * self._units["time"]).to(u.Myr).value
        self.t2 = 0

    def table(self, expr=None):
        if self._table is None:
            tbl = super(SgrSimulation, self).table()

            for x in "xyz":
                tbl[x].unit = self._units["length"]

            for x in ("vx","vy","vz"):
                tbl[x].unit = self._units["length"]/self._units["time"]

            self._table = tbl

        return super(SgrSimulation, self).table(expr=expr)

    def satellite(self, bound_expr="tub==0", frame="galactocentric",
                  column_names=None):
        s = super(SgrSimulation, self).satellite(bound_expr=bound_expr,
                                                 frame=frame,
                                                 column_names=column_names)
        s.meta["m"] = self.mass

        bound = self.table(bound_expr)
        s.meta["v_disp"] = np.sqrt(np.std(bound["vx"])**2 + \
                                   np.std(bound["vy"])**2 + \
                                   np.std(bound["vz"])**2)

        return s.decompose(usys)

    def particles(self, *args, **kwargs):
        p = super(SgrSimulation, self).particles(*args, **kwargs)
        return p.decompose(usys)
