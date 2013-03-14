# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

from streams.potential import *
from streams.simulation import Particle, TestParticleSimulation
from streams.data.gaia import parallax_error

def _variance_statistic(potential, xs, vs, sgr_cen):
    """ Compute the variance scalar that we will minimize.
        
        Parameters
        ----------
        potential : Potential
            The full Milky Way potential object.
        xs : ndarray
            An array of positions for all particles at all times
        vs : ndarray
            An array of velocities for all particles at all times
        sgr_cen : SgrCen
            Data for the Sgr satellite center, interpolated onto the
            time grid for our particles.
    """
    
    # Define tidal radius, escape velocity for satellite
    msat = 2.5E8 # M_sun
    sgr_orbital_radius = np.sqrt(sgr_cen.x**2 + sgr_cen.y**2 + sgr_cen.z**2)
    m_halo_enclosed = potential.params["v_halo"]**2 * sgr_orbital_radius/potential.params["_G"]
    mass_enclosed = potential.params["M_disk"] + potential.params["M_sph"] + m_halo_enclosed

    r_tides = sgr_orbital_radius * (msat / mass_enclosed)**(1./3)
    v_escs = np.sqrt(potential.params["_G"] * msat / r_tides)
    
    # N = number of particles
    N = xs.shape[1]
    
    # Distance to satellite center and total velocity
    d = np.sqrt((xs[:,:,0] - sgr_cen.x[:,np.newaxis].repeat(N, axis=1))**2 +
                (xs[:,:,1] - sgr_cen.y[:,np.newaxis].repeat(N, axis=1))**2 + 
                (xs[:,:,2] - sgr_cen.z[:,np.newaxis].repeat(N, axis=1))**2) / r_tides[:,np.newaxis].repeat(N, axis=1)
    
    v = np.sqrt((vs[:,:,0] - sgr_cen.vx[:,np.newaxis].repeat(N, axis=1))**2 +
                (vs[:,:,1] - sgr_cen.vy[:,np.newaxis].repeat(N, axis=1))**2 +
                (vs[:,:,2] - sgr_cen.vz[:,np.newaxis].repeat(N, axis=1))**2) / v_escs[:,np.newaxis].repeat(N, axis=1)
    
    idx = np.argmin(d**2 + v**2, axis=0)
    
    min_ds = []
    for ii,jj in enumerate(idx):
        min_ds.append(d[jj,ii])
    
    min_vs = []
    for ii,jj in enumerate(idx):
        min_vs.append(v[jj,ii])
    
    return np.var(min_ds) + np.var(min_vs)

def back_integrate(potential, sgr_snap, sgr_cen, dt):
    """ Given the particle snapshot information and a potential, integrate the particles
        backwards and return the variance scalar.
    """

    # Initialize particle simulation with full potential
    simulation = TestParticleSimulation(potential=potential)

    for ii in range(sgr_snap.num):
        p = Particle(position=(sgr_snap.data["x"][ii], sgr_snap.data["y"][ii], sgr_snap.data["z"][ii]), # kpc
                     velocity=(sgr_snap.data["vx"][ii], sgr_snap.data["vy"][ii], sgr_snap.data["vz"][ii]), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)

    ts, xs, vs = simulation.run(t1=max(sgr_cen.data["t"]), t2=min(sgr_cen.data["t"]), dt=-dt)

    return _variance_statistic(potential, xs, vs, sgr_cen)

def add_observational_uncertainty(x,y,z,vx,vy,vz):
    """ Given 3D galactocentric position and velocity, transform to heliocentric
        coordinates, apply observational uncertainty estimates, then transform
        back to galactocentric frame.
    """
    # Transform to heliocentric coordinates
    rsun = 8. # kpc
    
    x += rsun
    
    d = np.sqrt(x**2 + y**2 + z**2)
    vr = (x*vx + y*vy + z*vz) / d 
    
    # proper motions in km/s/kpc
    rad = np.sqrt(x**2 + y**2)
    vrad = (x*vx + y*vy) / rad
    mul = (x*vy - y*vx) / rad / d
    mub = (-z*vrad + rad*vz) / d**2
    
    # angular position
    sinb = z/d
    cosb = rad/d
    cosl = x/rad
    sinl = y/rad
    
    # DISTANCE ERROR -- assuming 2% distances from RR Lyrae mid-IR
    d += np.random.normal(0., 0.02*d)
    
    # VELOCITY ERROR -- 5 km/s (TODO: ???)
    vr += np.random.normal(0., (5.*u.km/u.s).to(u.kpc/u.Myr).value)
    
    dmu = proper_motion_error(d)
    
    # translate to radians/year
    conv1 = np.pi/180./60./60./1.e6
    # translate to km/s from  kpc/year 
    kmpkpc = 3.085678e16
    secperyr = 3.1536e7 
    conv2 = kmpkpc/secperyr
    dmu = dmu*conv1*conv2
    
    mul += np.random.normal(0., dmu)
    mub += np.random.normal(0., dmu)
    
    # CONVERT BACK
    x = d*cosb*cosl - rsun
    y = d*cosb*sinl
    z = d*sinb
    
    vx = vr*cosb*cosl - d*mul*sinl - d*mub*sinb*cosl
    vy = vr*cosb*sinl + d*mul*cosl - d*mub*sinb*sinl
    vz = vr*sinb + d*mub*cosb
    
    return (x,y,z,vx,vy,vz)

def back_integrate_with_errors(potential, sgr_snap, sgr_cen, dt):
    """ Given the particle snapshot information and a potential, integrate the particles
        backwards WITH "realistic" observational uncertainties.
    """

    # Initialize particle simulation with full potential
    simulation = TestParticleSimulation(potential=potential)
    
    # Distances in kpc, velocities in kpc/Myr
    x, y, z = sgr_snap.data["x"], sgr_snap.data["y"], sgr_snap.data["z"]
    vx, vy, vz = sgr_snap.data["vx"], sgr_snap.data["vy"], sgr_snap.data["vz"]
    
    x,y,z,vx,vy,vz = add_observational_uncertainty(x,y,z,vx,vy,vz)
    
    for ii in range(sgr_snap.num):
        p = Particle(position=(x[ii], y[ii], z[ii]), # kpc
                     velocity=(vx[ii], vy[ii] ,vz[ii]), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)

    ts, xs, vs = simulation.run(t1=max(sgr_cen.data["t"]), t2=min(sgr_cen.data["t"]), dt=-dt)
    
    return _variance_statistic(potential, xs, vs, sgr_cen)