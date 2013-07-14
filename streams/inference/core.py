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
    
    return (particle_orbits._r - satellite_orbit._r) / r_tide, \
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
    
    cov_matrix = np.cov(min_ps.T)
    
    # cov_matrix -> (6 x 6) covariance matrix for particles
    #w,v = np.linalg.eig(cov_matrix)
    #return np.sum(w)
    #return np.prod(w)
    
    sign,logdet = np.linalg.slogdet(cov_matrix)
    return -logdet