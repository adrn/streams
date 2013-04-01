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

__all__ = ["minimum_distance_matrix", "back_integrate", "generalized_variance"]

def minimum_distance_matrix(potential, xs, vs, sgr_cen):
    """ Compute the Nx6 matrix of minimum phase-space distance vectors.
        
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
    
    min_xs = np.zeros((N,3))
    for ii,jj in enumerate(idx):
        # xs[time, particles, dimension]
        min_xs[ii] = (xs[jj,ii] - sgr_cen.xyz[:,jj]) / r_tides[jj]
    
    min_vs = np.zeros((N,3))
    for ii,jj in enumerate(idx):
        # vs[time, particles, dimension]
        min_vs[ii] = (vs[jj,ii] - sgr_cen.vxyz[:,jj]) / v_escs[jj]
    
    min_ps = np.hstack((min_xs, min_vs))
    # min_ps -> (N x 6) matrix
    
    return min_ps
    
def generalized_variance(potential, xs, vs, sgr_cen):
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
    
    min_ps = minimum_distance_matrix(potential, xs, vs, sgr_cen)
    
    cov_matrix = np.cov(min_ps.T)
    # cov_matrix -> (6 x 6) covariance matrix for particles
    
    return np.linalg.det(cov_matrix)

def back_integrate(potential, sgr_snap, sgr_cen, dt):
    """ Given the particle snapshot information and a potential, integrate the particles
        backwards and return the variance scalar.
    """

    # Initialize particle simulation with full potential
    simulation = TestParticleSimulation(potential=potential)
    
    # Distances in kpc, velocities in kpc/Myr
    xyz = sgr_snap.xyz
    vxyz = sgr_snap.vxyz
    
    for ii in range(len(sgr_snap)):
        p = Particle(position=(xyz[0,ii].to(u.kpc).value, xyz[1,ii].to(u.kpc).value, xyz[2,ii].to(u.kpc).value), # kpc
                     velocity=(vxyz[0,ii].to(u.kpc/u.Myr).value, vxyz[1,ii].to(u.kpc/u.Myr).value, vxyz[2,ii].to(u.kpc/u.Myr).value), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)
    
    ts, xs, vs = simulation.run(t1=max(sgr_cen.t), t2=min(sgr_cen.t), dt=-dt)
    return ts, xs, vs
