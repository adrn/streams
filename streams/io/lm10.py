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
from .core import read_table, table_to_particles, table_to_orbits
from ..util import project_root, u_galactic
from ..dynamics import Particle, Orbit
from ..integrate.leapfrog import LeapfrogIntegrator
from ..potential.lm10 import LawMajewski2010

__all__ = ["particle_table", "particles_today", "satellite_today", \
           "time", "satellite_orbit"]

_lm10_path = os.path.join(project_root, "data", "simulation", "LM10")

class LM10Simulation(SimulationData):

    def __init__(self,
                 filename=os.path.join(project_root,"SgrTriax_DYN.dat")):
        """ ...

            Parameters
            ----------

        """
        super(LM10Simulation, self).__init__(...)
        self._hel_colnames = ("l","b","D","mul","mub","vr") # TODO: FIX

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


def satellite_today():
    """ Read in the position and velocity of the Sgr satellite at the end of
        the Law & Majewski 2010 simulation (e.g., present day position to be
        back-integrated).

    """

    # Here are the true parameters from the last block in R601LOG
    r0 = np.array([[2.3279727753E+01,2.8190329987,-6.8798148785]])*length_unit
    v0 = np.array([[3.9481694047,-6.1942673069E-01,3.4555581435]])*length_unit/time_unit

    satellite = Particle(r=r0.to(u.kpc),
                         v=v0.to(u.kpc/u.Myr),
                         m=[6E8]*u.M_sun)

    return satellite

def satellite_orbit():
    """ Read in the full orbit of the satellite. """
    sat_colnames = ["t","x","y","z","vx","vy","vz","col8","col9","col10"]
    data = read_table(filename="orb780.dat", expr="t<0", path=_lm10_path)

    oc = table_to_orbits(data, lm10_usys,
                         position_columns=["x","y","z"],
                         velocity_columns=["vx","vy","vz"])

    return oc.to(u_galactic)

def time():
    """ Time information for the Law & Majewski 2010 simulation """
    t1 = 0.
    t2 = -8000.

    return t1, t2
