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

# Project
from .core import read_simulation
from ...misc.units import UnitSystem
from ...nbody import ParticleCollection, OrbitCollection
from ...integrate.leapfrog import LeapfrogIntegrator
from ...potential.lm10 import LawMajewski2010

__all__ = ["lm10_particles", "lm10_particle_data", \
           "lm10_satellite", "lm10_satellite_orbit", "lm10_time"]    

def lm10_particle_data(N=None, expr=None):
    """ """
    
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
    
    if N != None and N > 0 and N < len(particle_data):
        idx = np.random.randint(0, len(particle_data), N)
        particle_data = particle_data[idx]
    
    return particle_data

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
    particle_data = lm10_particle_data(N=N, expr=expr)
    
    nparticles = len(particle_data)
    
    r = np.zeros((nparticles, 3))
    r[:,0] = np.array(particle_data["x"])
    r[:,1] = np.array(particle_data["y"])
    r[:,2] = np.array(particle_data["z"])
    
    v = np.zeros((nparticles, 3))
    v[:,0] = np.array(particle_data["vx"])
    v[:,1] = np.array(particle_data["vy"])
    v[:,2] = np.array(particle_data["vz"])
    
    usys = UnitSystem(u.kpc,u.Myr,u.M_sun)
    particles = ParticleCollection(r=r*u.kpc,
                                   v=v*u.km/u.s,
                                   m=np.zeros(len(r))*u.M_sun,
                                   unit_system=usys)
    
    return particles

def lm10_satellite():
    """ Read in the position and velocity of the Sgr satellite at the end of
        the Law & Majewski 2010 simulation (e.g., present day position to be
        back-integrated).
    
    """    
    # initial conditions, or, position of the satellite today from the text of
    #   the paper. i need to integrate these to the first timestep to get the
    #   true initial conditions for the satellite
    r0 = [[19., 2.7, -6.9]] # kpc
    v0 = ([[230., -35., 195.]]*u.km/u.s).to(u.kpc/u.Myr).value
    
    # get first timestep
    t1,t2 = lm10_time()
    
    # define true potential
    lm10 = LawMajewski2010()
    
    # integrate up to the first timestep
    integrator = LeapfrogIntegrator(lm10._acceleration_at, r0, v0)
    t,r,v = integrator.run(t1=0., t2=t1, dt=t1/10.)
    
    satellite = ParticleCollection(r=r[-1]*u.kpc, v=v[-1]*u.kpc/u.Myr, 
                                   m=[2.5E8]*u.M_sun)
    
    return satellite
    
def lm10_time():
    """ Read in the time information for the Law & Majewski 2010 simulation 
        (e.g., present day isn't exactly t=0)
        
    """
    sat_colnames = ["t","lambda_sun","beta_sun","ra","dec", \
                    "x_sun","y_sun","z_sun","x_gc","y_gc","z_gc", \
                    "dist","vgsr"]
    
    satellite_data = read_simulation(filename="SgrTriax_orbit.dat",
                                    column_names=sat_colnames)
    
    # they integrate past present day, so only select the prior history
    satellite_data = satellite_data[satellite_data["t"] <= 0.]
    
    t1 = max(satellite_data["t"])*1000.
    t2 = min(satellite_data["t"])*1000.
    
    return t1, t2

def lm10_satellite_orbit():
    """ Read in the full orbit of the satellite.
        
    """
    sat_colnames = ["t","lambda_sun","beta_sun","ra","dec", \
                    "x_sun","y_sun","z_sun","x_gc","y_gc","z_gc", \
                    "dist","vgsr"]
    
    col_map = dict(x_gc="x", y_gc="y", z_gc="z")
    col_scales = dict(x=-1.)
    
    satellite_data = read_simulation(filename="SgrTriax_orbit.dat",
                                    column_names=sat_colnames,
                                    column_map=col_map,
                                    column_scales=col_scales)
    
    # they integrate past present day, so only select the prior history
    satellite_data = satellite_data[satellite_data["t"] <= 0.]
    
    r = np.zeros((len(satellite_data), 1, 3))
    r[:,0,0] = np.array(satellite_data["x"])
    r[:,0,1] = np.array(satellite_data["y"])
    r[:,0,2] = np.array(satellite_data["z"])
    
    v = np.zeros_like(r)
    
    usys = UnitSystem(u.kpc,u.Myr,u.M_sun)    
    return OrbitCollection(t=(satellite_data['t']*u.Gyr).to(u.Myr),
                           r=r*u.kpc, v=v*u.km/u.s, 
                           unit_system=usys)
                           