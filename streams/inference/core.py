# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

from ..potential import *

__all__ = ["minimum_distance_matrix",
           "relative_normalized_coordinates",
           "generalized_variance"]
    
def relative_normalized_coordinates(potential, particle_orbits, satellite_orbit):
    """ Compute the coordinates of particles relative to the satellite, 
        with positions normalized by the tidal radius and velocities
        normalized by the escape velocity. 
        
        Note::
            Assumes the particle and satellite position/velocity 
            units are the same!
        
        Parameters
        ----------
        potential : streams.CartesianPotential
        particle_orbits : OrbitCollection
        satellite_orbit : OrbitCollection
    """
    
    # need to add a new axis to normalize each coordinate component
    r_tide = potential._tidal_radius(m=satellite_orbit._m,
                                     r=satellite_orbit._r)[:,:,np.newaxis]
    v_esc = potential._escape_velocity(m=satellite_orbit._m,
                                       r_tide=r_tide)
    
    return (particle_orbits._r - satellite_orbit._r) / (r_tide), \
           (particle_orbits._v - satellite_orbit._v) / v_esc

def minimum_distance_matrix(potential, particle_orbits, satellite_orbit):
    """ Compute the Nx6 matrix of minimum phase-space distance vectors.
        
        Parameters
        ----------
        potential : Potential
        particle_orbits : OrbitCollection
        satellite_orbit : OrbitCollection
    """
    
    Nparticles = particle_orbits._r.shape[1]
    
    R,V = relative_normalized_coordinates(potential, particle_orbits, satellite_orbit) 
    #D_R = np.fabs(np.sqrt(np.sum(R**2, axis=-1)) - 1.)
    #D_ps = np.sqrt(D_R**2. + np.sum(V**2, axis=-1))
    D_ps = np.sqrt(np.sum(R**2, axis=-1) + np.sum(V**2, axis=-1))
    
    # Find the index of the time of the minimum D_ps for each particle
    min_time_idx = D_ps.argmin(axis=0)
    
    min_ps = np.zeros((Nparticles,6))
    xx = zip(min_time_idx, range(Nparticles))
    for kk in range(Nparticles):
        jj,ii = xx[kk]
        min_ps[ii] = np.append(R[jj,ii], V[jj,ii])
    
    return min_ps

def generalized_variance(potential, particle_orbits, satellite_orbit):
    """ Compute the variance scalar that we will minimize.
        
        Parameters
        ----------
        potential : Potential
            The full Milky Way potential object.
        particle_orbits : OrbitCollection
            An object containing orbit information for a collection of 
            particles.
        satellite_orbit : OrbitCollection
            Data for the Sgr satellite center, interpolated onto the
            time grid for our particles.
    """
    
    min_ps = minimum_distance_matrix(potential, particle_orbits, satellite_orbit)
    cov_matrix = np.cov(np.fabs(min_ps.T))
    sign,logdet = np.linalg.slogdet(cov_matrix)
    return logdet**2

def objective(potential, satellite_orbit, particle_orbits, v_disp):
    """ This is a new objective function, motivated by the fact that what 
        I was doing before doesn't really make sense...
    """
    
    # get numbers for any relevant loops below
    Ntimesteps, Nparticles, Ndim = particle_orbits._r.shape
    
    r_tide = potential._tidal_radius(m=satellite_orbit._m,
                                     r=satellite_orbit._r)
    v_esc = potential._escape_velocity(m=satellite_orbit._m,
                                       r_tide=r_tide)
    r_tide = r_tide[:,:,np.newaxis]
    v_esc = v_esc[:,:,np.newaxis]
    
    # compute relative, normalized coordinates and then phase-space distance
    R = particle_orbits._r - satellite_orbit._r
    V = particle_orbits._v - satellite_orbit._v
    Q = R / r_tide
    P = V / v_esc
    D_ps = np.sqrt(np.sum(Q**2, axis=-1) + np.sum(P**2, axis=-1))
    
    # Find the index of the time of the minimum D_ps for each particle
    min_time_idx = D_ps.argmin(axis=0)
    cov = np.zeros((6,6))
    b = np.vstack((R.T, V.T)).T
    for ii in range(Nparticles):
        idx = min_time_idx[ii]
        r_disp = np.squeeze(r_tide[idx])
        c = b[idx,ii] / np.array([r_disp]*3+[v_disp]*3)
        cov += np.outer(c, c.T)
    cov /= Nparticles
    
    sign,logdet = np.linalg.slogdet(cov)
    return logdet

def objective2(potential, satellite_orbit, particle_orbits, v_disp):
    """ This is a new objective function, motivated by the fact that what 
        I was doing before doesn't really make sense...
    """
    
    # get numbers for any relevant loops below
    Ntimesteps, Nparticles, Ndim = particle_orbits._r.shape
    
    r_tide = potential._tidal_radius(m=satellite_orbit._m,
                                     r=satellite_orbit._r)
    
    # compute relative, normalized coordinates and then phase-space distance
    R = particle_orbits._r - satellite_orbit._r
    V = particle_orbits._v - satellite_orbit._v
    
    full_r_tide = np.repeat(r_tide, 3, axis=1)
    full_v_disp = np.zeros_like(full_r_tide) + v_disp
    cov = np.hstack((full_r_tide, full_v_disp))**2
    
    L = []
    for jj in range(Nparticles):
        X = np.hstack((R[:,jj], V[:,jj]))
        fac = np.prod(cov,axis=1)**-0.5
        a = -0.5 * np.sum(X**2 / cov, axis=1)
        
        A = np.max(a)
        l = A + np.log(np.sum(fac * np.exp(a-A)))
        L.append(l)
    
    return np.sum(L)
    
    #
    ##
    #
    
    D_ps = np.sum(Q**2, axis=-1) + np.sum(P**2, axis=-1)
    return np.sum(np.min(D_ps, axis=0))
    
    min_time_idx = D_ps.argmin(axis=0)
    print(min_time_idx[0])
    print(np.sum(Q**2, axis=-1)[min_time_idx[0],0], np.sum(P**2, axis=-1)[min_time_idx[0],0])
    sys.exit(0)
    
    B = 0.
    for ii in range(Nparticles):
        idx = min_time_idx[ii]
        r_disp = np.squeeze(r_tide[idx])        
        B += np.log(np.prod([r_disp**2]*3+[v_disp**2]*3)) + D_ps[idx,ii]
    
    return B