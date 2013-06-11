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
from ..nbody import ParticleCollection, OrbitCollection

__all__ = ["orphan_particles", "orphan_satellite", "orphan_time"]

X = (G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value / 0.15**3 * 1E7)**-0.5
length_unit = u.Unit("0.15 kpc")
mass_unit = u.Unit("1E7 M_sun")
time_unit = u.Unit("{:08f} Myr".format(X))

def orphan_particles(N=None, expr=None):
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
    particle_filename = os.path.join(project_root, 
                                     "data",
                                     "simulation",
                                     "Orphan",
                                     "ORP_SNAP")
    particle_colnames = ["m","x","y","z","vx","vy","vz","s1","s2","tub"]
    
    particle_data = ascii.read(particle_filename, names=particle_colnames)
    
    if expr != None and len(expr.strip()) > 0:
        idx = numexpr.evaluate(str(expr), particle_data)
        particle_data = particle_data[idx]
    
    if N != None and N > 0:
        idx = np.random.randint(0, len(particle_data), N)
        particle_data = particle_data[idx]
    
    r = np.zeros((len(particle_data), 3))
    r[:,0] = np.array(particle_data["x"])
    r[:,1] = np.array(particle_data["y"])
    r[:,2] = np.array(particle_data["z"])
    r = r*length_unit
    
    v = np.zeros((len(particle_data), 3))
    v[:,0] = np.array(particle_data["vx"])
    v[:,1] = np.array(particle_data["vy"])
    v[:,2] = np.array(particle_data["vz"])
    v = v*length_unit/time_unit
    
    particles = ParticleCollection(r=r.to(u.kpc),
                                   v=v.to(u.km/u.s),
                                   m=np.zeros(len(r))*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    return particles

def orphan_satellite():
    """ Read in the position and velocity of the Orphan satellite center 
        at the end of the simulation (e.g., present day position to be
        back-integrated).
    
    """
    sat_filename = os.path.join(project_root, 
                                "data",
                                "simulation", 
                                "Orphan",
                                "ORP_CEN")
    sat_colnames = ["t","dt","x","y","z","vx","vy","vz"]
    
    # read in ascii file
    satellite_data = ascii.read(sat_filename, names=sat_colnames)
    
    # initial conditions, or, position of the satellite today
    i = -1
    r = [[satellite_data[i]['x'], 
          satellite_data[i]['y'],
          satellite_data[i]['z']]]
    r = r*length_unit
    
    v = [[satellite_data[i]['vx'], 
          satellite_data[i]['vy'],
          satellite_data[i]['vz']]]
    v = v*length_unit/time_unit
    
    satellite = ParticleCollection(r=r.to(u.kpc), 
                                   v=v.to(u.km/u.s), 
                                   m=1E7*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    return satellite
    
def orphan_time():
    """ Read in the time information for Orphan stream simulation 
    """
    sat_filename = os.path.join(project_root, 
                                "data",
                                "simulation", 
                                "Orphan",
                                "ORP_CEN")
    sat_colnames = ["t","dt","x","y","z","vx","vy","vz"]
    
    # read in ascii file
    satellite_data = ascii.read(sat_filename, names=sat_colnames)
    
    t1 = (max(satellite_data["t"])*time_unit).to(u.Myr).value
    t2 = (min(satellite_data["t"])*time_unit).to(u.Myr).value
    
    return t1, t2
