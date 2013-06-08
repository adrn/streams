# coding: utf-8

""" Classes for accessing simulation data related to Sagittarius. """

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

# Project
from ..misc.units import UnitSystem
from ..util import project_root
from ..nbody import Particle, ParticleCollection, Orbit, OrbitCollection
from ..integrate.leapfrog import LeapfrogIntegrator
from ..potential.lm10 import LawMajewski2010

__all__ = ["lm10_particles", "lm10_satellite", "lm10_time"]

def read_simulation(filename, column_names, column_map=dict(), 
                    column_scales=dict(), path=None):
    """ Read in simulation data from an ASCII file. 
    
        Parameters
        ----------
        filename : str
        column_names : list
        column_map : dict (optional)
        path : str (optional)
    """
    
    if path is None:
        path = os.path.join(project_root, "data", "simulation")
    
    full_path = os.path.join(path, filename)
    
    # use astropy.io.ascii to read the ascii data
    data = ascii.read(full_path, names=column_names)
    
    # use the column map to rename columns
    for old_name,new_name in column_map.items():
        data.rename_column(old_name, new_name)
    
    # rescale columns using specified dict
    for name, scale in column_scales.items():
        data[name] = scale*data[name]
    
    return data
    

def lm10_particles(N=None, expr=None):
    """ Read in particles from the Law & Majewski 2010 simulation of Sgr. 
    
        Parameters
        ----------
        N : int
            Number of particles to read. None means 'all'.
        expr : str
            String selection condition to be fed to numexpr for selecting 
            particles.
        
    """
    particle_colnames = ["Lambda", "Beta", "ra", "dec", "l", "b", \
                         "xgc", "ygc", "zgc", "xsun", "ysun", "zsun", \
                         "x4", "y4", "z4", "u", "v", "w", "dist", "vgsr", \
                         "mul", "mub", "mua", "mud", "Pcol", "Lmflag"]
    
    col_map = dict(xgc="x", ygc="y", zgc="z", u="vx", v="vy", w="vz")
    col_scales = dict(x=-1., vx=-1.)
    
    particle_data = read_simulation(filename="SgrTriax_DYN.dat",
                                    column_names=particle_colnames,
                                    column_map=col_map,
                                    column_scales=col_scales)
    
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
    
    v = np.zeros((len(particle_data), 3))
    v[:,0] = np.array(particle_data["vx"])
    v[:,1] = np.array(particle_data["vy"])
    v[:,2] = np.array(particle_data["vz"])
    
    usys = UnitSystem(u.kpc,u.Myr,u.M_sun)
    particles = ParticleCollection(r=r*u.kpc,
                                   v=v*u.km/u.s,
                                   m=np.zeros(len(r))*u.M_sun,
                                   units=usys)
    
    return particles

def lm10_satellite():
    """ Read in the position and velocity of the Sgr satellite at the end of
        the Law & Majewski 2010 simulation (e.g., present day position to be
        back-integrated).
    
    """
    sat_filename = os.path.join(project_root, 
                                "data",
                                "simulation", 
                                "SgrTriax_orbit.dat")
    sat_colnames = ["t","lambda_sun","beta_sun","ra","dec", \
                    "x_sun","y_sun","z_sun","x_gc","y_gc","z_gc", \
                    "dist","vgsr"]
    
    # read in ascii file
    satellite_data = ascii.read(sat_filename, names=sat_colnames)
    
    # they integrate past present day, so only select the prior history
    satellite_data = satellite_data[satellite_data["t"] <= 0.]
    
    # convert column names to my own convention
    satellite_data.rename_column("lambda_sun", "lambda")
    satellite_data.rename_column("beta_sun", "beta")
    satellite_data.rename_column("x_gc", "x")
    satellite_data.rename_column("y_gc", "y")
    satellite_data.rename_column("z_gc", "z")
    
    # they use a left-handed coordinate system?
    satellite_data["x"] = -satellite_data["x"]
    
    # initial conditions, or, position of the satellite today from the text of
    #   the paper. i need to integrate these to the first timestep to get the
    #   true initial conditions for the satellite
    r0 = [[19., 2.7, -6.9]] # kpc
    v0 = ([[230., -35., 195.]]*u.km/u.s).to(u.kpc/u.Myr).value
    
    # get first timestep
    t1 = max(satellite_data["t"])*1000.
    dt = t1
    
    # define true potential
    lm10 = LawMajewski2010()
    
    # integrate up to the first timestep
    integrator = LeapfrogIntegrator(lm10._acceleration_at, r0, v0)
    t,r,v = integrator.run(time_spec=dict(t1=0., t2=t1, dt=dt))
    
    satellite = ParticleCollection(r=r[1]*u.kpc, v=v[1]*u.kpc/u.Myr, 
                                   m=[2.5E8]*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    return satellite
    
def lm10_time():
    """ Read in the time information for the Law & Majewski 2010 simulation 
        (e.g., present day isn't exactly t=0)
        
    """
    sat_filename = os.path.join(project_root, 
                                "data",
                                "simulation", 
                                "SgrTriax_orbit.dat")
    sat_colnames = ["t","lambda_sun","beta_sun","ra","dec", \
                    "x_sun","y_sun","z_sun","x_gc","y_gc","z_gc", \
                    "dist","vgsr"]
    
    # read in ascii file
    satellite_data = ascii.read(sat_filename, names=sat_colnames)
    
    # they integrate past present day, so only select the prior history
    satellite_data = satellite_data[satellite_data["t"] <= 0.]
    
    t1 = max(satellite_data["t"])*1000.
    t2 = min(satellite_data["t"])*1000.
    
    return t1, t2