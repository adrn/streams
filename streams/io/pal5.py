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
                          "gaia_challenge", "pal5")
_particle_file = "pal5.all_particles.txt"

# This is used for the SgrTriax*.dat files
usys = UnitSystem(u.kpc, u.M_sun, u.Myr)

def particle_table(N=None, expr=None):
    """ Read in particles from a specified satellite.
    
        Parameters
        ----------
        N : int
            Number of particles to read. None means 'all'.
        expr : str
            String selection condition to be fed to numexpr for selecting 
            particles.
        
    """
    
    data = ascii.read(os.path.join(_data_path, _particle_file))
    
    if N != None and N > 0:
        idx = np.random.randint(0, len(data), N)
        data = data[idx]
    
    return data

def particles_today(N=None, expr=None):
    """ Read in particles from a specified satellite.
    
        Parameters
        ----------
        N : int
            Number of particles to read. None means 'all'.
        expr : str
            String selection condition to be fed to numexpr for selecting 
            particles.
    """
    data = particle_table(N=N, expr=expr)
    
    data['x'] = (np.array(data['x'])*u.pc).to(u.kpc).value
    data['y'] = (np.array(data['y'])*u.pc).to(u.kpc).value
    data['z'] = (np.array(data['z'])*u.pc).to(u.kpc).value
    
    data['vx'] = (np.array(data['vx'])*u.km/u.s).to(u.kpc/u.Myr).value
    data['vy'] = (np.array(data['vy'])*u.km/u.s).to(u.kpc/u.Myr).value
    data['vz'] = (np.array(data['vz'])*u.km/u.s).to(u.kpc/u.Myr).value
    
    pc = table_to_particles(data, usys,
                            position_columns=["x","y","z"],
                            velocity_columns=["vx","vy","vz"])
    
    return pc.to(UnitSystem.galactic())

def satellite_today():
    """ From Andreas """
    
    # Here are the true parameters from the last block in R601LOG
    r0 = np.array([[7.816082584,.240023507,16.640055966]])*u.kpc
    v0 = np.array([[-37.456858,-151.794112,-21.609662]])*u.km/u.s
    
    satellite = ParticleCollection(r=r0.to(u.kpc), 
                                   v=v0.to(u.kpc/u.Myr), 
                                   m=[2E4]*u.M_sun)
    
    return satellite

def time():
    """ Time information """
    t1 = 0.
    t2 = -4200.
    
    return t1, t2
