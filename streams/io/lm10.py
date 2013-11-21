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
from .core import SimulationData
from ..util import project_root

__all__ = ["LM10Simulation"]

_lm10_path = os.path.join(project_root, "data", "simulation", "LM10")

class LM10Simulation(SimulationData):

    def __init__(self,
                 filename=os.path.join(_lm10_path,"SgrTriax_DYN.dat")):
        """ ...

            Parameters
            ----------

        """
        super(LM10Simulation, self).__init__(...)
        self._hel_colnames = ("l","b","D","mul","mub","vr") # TODO: FIX
        self.t1 = 0.
        self.t2 = -8000.

    @property
    def table(self):
        tbl = super(LM10Simulation, self).table
        tbl.rename_column("xgc","x")
        tbl.rename_column("ygc","y")
        tbl.rename_column("zgc","z")

        tbl.rename_column("u","vx")
        tbl.rename_column("v","vy")
        tbl.rename_column("w","vz")

        tbl["x"] = -tbl["x"]
        tbl["vx"] = -tbl["vx"]

        for x in "xyz":
            tbl[x].unit = u.kpc

        for x in ("vx","vy","vz"):
            tbl[x].unit = u.km/u.s

        return tbl

    def satellite(self, bound_expr="Pcol>-1", frame="galactocentric",
                  column_names=None):
        super(LM10Simulation, self).satellite(bound_expr=bound_expr,
                                              frame=frame,
                                              column_names=column_names)
