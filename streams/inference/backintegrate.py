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

__all__ = ["tidal_radius", "escape_velocity",
           "minimum_distance_matrix",
           "relative_normalized_coordinates",
           "generalized_variance"]


def tidal_radius(potential, r, m_sat=2.5E8):
    """ Compute the tidal radius of the given satellite over the orbit 
        or position provided.
        
        Parameters
        ----------
        potential : streams.CartesianPotential
        satellite_orbit : streams.Orbit
    """
    
    # assume the satellite has the right units already...
    
    # Radius of Sgr center relative to galactic center
    R_orbit = np.sqrt(np.sum(r**2., axis=-1)) 
    
    _G = 4.4997533243534949e-12 # kpc^3 / Myr^2 / M_sun
    
    m_halo_enc = potential["halo"]._parameters["v_halo"]**2 * R_orbit/_G
    m_enc = potential["disk"]._parameters["m"] + \
            potential["bulge"]._parameters["m"] + \
            m_halo_enc
    
    return R_orbit * (m_sat / m_enc)**(1./3)

def escape_velocity(potential, r_tide, m_sat):
    """ Compute the escape velocity of a satellite in a potential given
        its tidal radius.
        
        Parameters
        ----------
        potential : streams.CartesianPotential
        r_tide : ndarray
    
    """
    
    _G = 4.4997533243534949e-12 # kpc^3 / Myr^2 / M_sun
    
    return np.sqrt(_G * m_sat / r_tide)
    
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
    r_tide = tidal_radius(potential, satellite_orbit._r)[:,:,np.newaxis]
    v_esc = escape_velocity(potential, r_tide, m_sat=satellite_orbit.m.value)
    
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
    w,v = np.linalg.eig(cov_matrix)
    
    return np.sum(w)

