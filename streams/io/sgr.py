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
from .core import read_simulation
from ...misc.units import UnitSystem
from ...nbody import ParticleCollection, OrbitCollection
from ...integrate.leapfrog import LeapfrogIntegrator
from ...potential.lm10 import LawMajewski2010

__all__ = ["lm10_particles", "lm10_particle_data", \
           "lm10_satellite", "lm10_satellite_orbit", "lm10_time"]    

_lm10_path = os.path.join(project_root, "data", "simulation", "LM10")

# These are used for the orbit file David Law sent me
X = (G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value / 0.85**3 * 6.4E8)**-0.5
length_unit = u.Unit("0.85 kpc")
mass_unit = u.Unit("6.4E8 M_sun")
time_unit = u.Unit("{:08f} Myr".format(X))

# This is used for the SgrTriax*.dat files
lm10_usys = UnitSystem(u.kpc, u.M_sun, u.Gyr)

def particles(N=None, expr=None):
    """ Read in particles from Law & Majewski 2010.
    
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
    col_map = dict(xgc="x", ygc="y", zgc="z", u="vx", v="vy", w="vz")
    col_scales = dict(x=-1., vx=-1.)
    data = read_table("SgrTriax_DYN.dat", path=_lm10_path, 
                      column_map=col_map, column_scales=col_scales,
                      N=N, expr=expr)
    
    pc = table_to_particles(data, lm10_usys,
                            position_columns=["x","y","z"],
                            velocity_columns=["vx","vy","vz"])
    
    return pc.to(UnitSystem.galactic())

def satellite():
    """ Read in the position and velocity of the Sgr satellite at the end of
        the Law & Majewski 2010 simulation (e.g., present day position to be
        back-integrated).
    
    """    
    
    # Here are the true parameters from the last block in R601LOG
    r0 = np.array([[2.3279727753E+01,2.8190329987,-6.8798148785]])*length_unit
    v0 = np.array([[3.9481694047,-6.1942673069E-01,3.4555581435]])*length_unit/time_unit
    
    satellite = ParticleCollection(r=r0.to(u.kpc), 
                                   v=v0.to(u.kpc/u.Myr), 
                                   m=[2.5E8]*u.M_sun)
    
    return satellite
    
def time():
    """ Time information for the Law & Majewski 2010 simulation """    
    t1 = 0.
    t2 = -8000.
    
    return t1, t2
