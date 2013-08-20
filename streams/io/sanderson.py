# coding: utf-8

""" Classes for accessing simulation data for the Gaia Data Challenge. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import numexpr
import astropy.units as u
from astropy.constants import G
from astropy.io import ascii

# Project
from .core import read_table, table_to_particles, table_to_orbits
from ..util import project_root
from ..misc import UnitSystem
from ..dynamics import ParticleCollection, OrbitCollection
from ..integrate.leapfrog import LeapfrogIntegrator

__all__ = ["particle_table", "particles_today", "satellite_today", "time", "satellite_orbit"]

_data_path = os.path.join(project_root, "data", "simulation", 
                          "gaia_challenge", "spherical")
_particle_file = "MilkyWay1.ne.txt"

# This is used for the SgrTriax*.dat files
usys = UnitSystem(u.kpc, u.M_sun, u.Myr)

def particle_table(N=None, expr=None, satellite_id=0):
    """ Read in particles from a specified satellite.
    
        Parameters
        ----------
        N : int
            Number of particles to read. None means 'all'.
        expr : str
            String selection condition to be fed to numexpr for selecting 
            particles.
        satellite_id : int
            The number of the satellite to read in.
        
    """
    
    # read index of satellite id
    with open(os.path.join(_data_path, "MilkyWay1.sats.txt")) as f:
        satellite_indices = f.readlines()[1:]
    
    # offset because first line is number of lines...damn IDL
    particle_index1 = int(satellite_indices[satellite_id]) + 1
    particle_index2 = int(satellite_indices[satellite_id+1]) + 1
    
    data = ascii.read(os.path.join(_data_path, _particle_file), 
                      data_start=particle_index1, data_end=particle_index2,
                      names=['x','y','z','vx','vy','vz'])
    
    return data

def particles_today(N=None, expr=None, satellite_id=0):
    """ Read in particles from a specified satellite.
    
        Parameters
        ----------
        N : int
            Number of particles to read. None means 'all'.
        expr : str
            String selection condition to be fed to numexpr for selecting 
            particles.
        satellite_id : int
            The number of the satellite to read in.
    """
    data = particle_table(N=N, expr=expr, satellite_id=satellite_id)
    
    data['vx'] = (np.array(data['vx'])*u.km/u.s).to(u.kpc/u.Myr).value
    data['vy'] = (np.array(data['vy'])*u.km/u.s).to(u.kpc/u.Myr).value
    data['vz'] = (np.array(data['vz'])*u.km/u.s).to(u.kpc/u.Myr).value
    
    pc = table_to_particles(data, usys,
                            position_columns=["x","y","z"],
                            velocity_columns=["vx","vy","vz"])
    
    return pc.to(UnitSystem.galactic())

def satellite_today():
    """ Read in the position and velocity of the Sgr satellite at the end of
        the Law & Majewski 2010 simulation (e.g., present day position to be
        back-integrated).
    
    """    
    
    # Here are the true parameters from the last block in R601LOG
    r0 = np.array([[2.3279727753E+01,2.8190329987,-6.8798148785]])*length_unit
    v0 = np.array([[3.9481694047,-6.1942673069E-01,3.4555581435]])*length_unit/time_unit
    
    satellite = ParticleCollection(r=r0.to(u.kpc), 
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
    
    return oc.to(UnitSystem.galactic())
    
def time():
    """ Time information for the Law & Majewski 2010 simulation """    
    t1 = 0.
    t2 = -9000.
    
    return t1, t2
