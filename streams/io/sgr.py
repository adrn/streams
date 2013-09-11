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
from .core import read_table, table_to_particles, table_to_orbits
from ..util import project_root
from ..dynamics import ParticleCollection, OrbitCollection
from ..misc import UnitSystem

__all__ = ["particles_today", "satellite_orbit", "satellite_today", "time"]

_path = os.path.join(project_root, "data", "simulation", "Sgr")

def usys_from_file(scfpar):
    """ Generate a unit system from an SCFPAR file. """
    
    with open(scfpar) as f:
        lines = f.readlines()
        length = float(lines[16].split()[0])
        mass = float(lines[17].split()[0])
    
    X = (G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value / length**3 * mass)**-0.5
    length_unit = u.Unit("{0} kpc".format(length))
    mass_unit = u.Unit("{0} M_sun".format(mass))
    time_unit = u.Unit("{:08f} Myr".format(X))
    usys = UnitSystem(length_unit, mass_unit, time_unit)
    
    return usys

def particles_today(m, N=None, expr=None):
    """ Read in particles from Kathryn's run of Sgr-like orbits for a 
        variety of masses.
    
        Parameters
        ----------
        N : int
            Number of particles to read. None means 'all'.
        expr : str
            String selection condition to be fed to numexpr for selecting 
            particles.
        
    """
    
    _full_path = os.path.join(_path, m)
    usys = usys_from_file(os.path.join(_full_path, "SCFPAR"))
    
    # Read in particle data -- a snapshot of particle positions, velocities at
    #   the end of the simulation
    data = read_table("SNAP", path=_full_path, N=N, expr=expr)
    pc = table_to_particles(data, usys,
                            position_columns=["x","y","z"],
                            velocity_columns=["vx","vy","vz"])
    
    return pc.to(UnitSystem.galactic())

def satellite_orbit(m):
    """ Read in the position and velocity of the Orphan satellite center 
        over the whole simulation.
    """
    
    _full_path = os.path.join(_path, m)
    usys = usys_from_file(os.path.join(_full_path, "SCFPAR"))
    
    data = read_table("SCFCEN", path=_full_path)
    orbit = table_to_orbits(data, usys,
                            position_columns=["x","y","z"],
                            velocity_columns=["vx","vy","vz"])
    
    return orbit

def satellite_today(m):
    """ Read in the position and velocity of the Orphan satellite center 
        at the end of the simulation (e.g., present day position to be
        back-integrated).
    
    """
    return satellite_orbit(m)[-1]
    
def time(m):
    """ Read in the time information for Orphan stream simulation 
    """
    s = satellite_orbit(m)
    
    t1 = np.max(s.t).to(u.Myr).value
    t2 = np.min(s.t).to(u.Myr).value
    
    return t1, t2
