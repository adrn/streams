# coding: utf-8

""" Classes for accessing simulation data for the Orphan stream. """

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
from ..util import project_root
from .core import read_table
from ..nbody import ParticleCollection, OrbitCollection
from ..misc import UnitSystem

__all__ = ["particles", "orphan_satellite", "orphan_time"]

_orphan_path = os.path.join(project_root, "data", "simulation", "Orphan")

X = (G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value / 0.15**3 * 1E7)**-0.5
length_unit = u.Unit("0.15 kpc")
mass_unit = u.Unit("1E7 M_sun")
time_unit = u.Unit("{:08f} Myr".format(X))
orphan_usys = UnitSystem(length_unit, mass_unit, time_unit)

def particles(N=None, expr=None):
    """ Read in particles from Kathryn's run of a satellite similar to 
        the progenitor of the Orphan stream
    
        Parameters
        ----------
        N : int
            Number of particles to read. None means 'all'.
        expr : str
            String selection condition to be fed to numexpr for selecting 
            particles.
        
    """
    
    # Read in particle data -- a snapshot of particle positions, velocities at
    #   the end of the simulation
    data = read_table("ORP_SNAP", path=_orphan_path, N=N, expr=expr)
    pc = table_to_particles(data, orphan_usys,
                            position_columns=["x","y","z"],
                            velocity_columns=["vx","vy","vz"])
    
    return pc.to(UnitSystem.galactic())

def satellite():
    """ Read in the position and velocity of the Orphan satellite center 
        at the end of the simulation (e.g., present day position to be
        back-integrated).
    
    """
    data = read_table("ORP_CEN", path=_orphan_path)
    orbit = table_to_orbits(data, orphan_usys,
                            position_columns=["x","y","z"],
                            velocity_columns=["vx","vy","vz"])
    
    return orbit[-1]
    
def time():
    """ Read in the time information for Orphan stream simulation 
    """
    data = read_table("ORP_CEN", path=_orphan_path)
    
    t1 = (max(data["t"])*time_unit).to(u.Myr).value
    t2 = (min(data["t"])*time_unit).to(u.Myr).value
    
    return t1, t2
