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

from streams.potential import *
from streams.simulation import TestParticle

__all__ = ["relative_normalized_coordinates",
           "phase_space_distance",
           "minimum_distance_matrix", 
           "generalized_variance"]

def relative_normalized_coordinates(particle_orbits, satellite_orbit, 
                                    r_tide, v_esc):
    """ Compute the coordinates of particles relative to the satellite, 
        with positions normalized by the tidal radius and velocities
        normalized by the escape velocity.
        
        Parameters
        ----------
        particle_orbits : TestParticleOrbit
        satellite_orbit : TestParticleOrbit
        r_tide : astropy.units.Quantity
        v_esc : astropy.units.Quantity
    """
    # THE BELOW IS A HACK HACK HACK BECAUSE OF:
    #   https://github.com/astropy/astropy/issues/974  
    #if not (particle_orbits.t == satellite_orbit.t).all():
    
    if not (particle_orbits.t.value == satellite_orbit.t.value).all():
        raise ValueError("Interpolation not yet supported. Time vectors must "
                         "be aligned.")
        
    if particle_orbits.r.value.ndim == 2:
        sat_r = satellite_orbit.r
        sat_v = satellite_orbit.v
        r_tide = r_tide[:,np.newaxis]
        v_esc = v_esc[:,np.newaxis]
    elif particle_orbits.r.value.ndim == 3:
        sat_r = satellite_orbit.r[:,np.newaxis,:]
        sat_v = satellite_orbit.v[:,np.newaxis,:]
        r_tide = r_tide[:,np.newaxis,np.newaxis]
        v_esc = v_esc[:,np.newaxis,np.newaxis]
    else:
        raise ValueError("")
    
    # THE BELOW IS A HACK HACK HACK BECAUSE OF:
    #   https://github.com/astropy/astropy/issues/974  
    sat_r = u.Quantity(sat_r.value, str(sat_r.unit))
    sat_v = u.Quantity(sat_v.value, str(sat_v.unit))
    
    return (particle_orbits.r - sat_r) / r_tide, (particle_orbits.v - sat_v) / v_esc
    
def phase_space_distance(R,V):
    """ Compute the phase-space distance from relative, normalized coordinates.
        
        Parameters
        ----------
        R : Quantity, array
            Positions of particles relative to satellite, normalized by R_tide.
        V : Quantity, array
            Velocities of particles relative to satellite, normalized by V_esc.
    """
    try:
        R = R.decompose().value
    except AttributeError:
        pass
    
    try:
        V = V.decompose().value
    except AttributeError:
        pass
    
    if R.shape != V.shape:
        raise ValueError("Shape of R,V must match!")
    
    if R.ndim != 3 or V.ndim != 3:
        raise ValueError("R and V should have shape: (Ntimesteps, Nparticles, 6)")
    
    if isinstance(R, u.Quantity):
        dist = np.sqrt(np.sum(R.decompose()**2,axis=2) + np.sum(V.decompose()**2, axis=2))
    else:
        dist = np.sqrt(np.sum(R**2,axis=2) + np.sum(V**2, axis=2))
        
    return dist

def minimum_distance_matrix(potential, particle_orbits, satellite_orbit):
    """ Compute the Nx6 matrix of minimum phase-space distance vectors.
        
        Parameters
        ----------
        potential : Potential
        particle_orbits : TestParticleOrbit
            ...
        satellite_orbit : TestParticleOrbit
            ...
    """
    
    Nparticles = particle_orbits.r.value.shape[1]
    unit_bases = potential.units.values()
    
    # Define tidal radius, escape velocity for satellite
    m_sat = 2.5E8 * u.M_sun
    
    # ----------------------------
    # THE BELOW IS A HACK HACK HACK BECAUSE OF:
    #   https://github.com/astropy/astropy/issues/974
    
    # Radius of Sgr center relative to galactic center
    #R_sgr = np.sqrt(satellite_orbit.r[:,0]**2 + \
    #                satellite_orbit.r[:,1]**2 + \
    #                satellite_orbit.r[:,2]**2) * satellite_orbit.r.unit
    
    #m_halo_enc = potential["halo"]._unscaled_parameters["v_halo"]**2 * R_sgr/G
    #m_enc = potential["disk"]._unscaled_parameters["m"] + \
    #        potential["bulge"]._unscaled_parameters["m"] + \
    #        m_halo_enc
    
    # Radius of Sgr center relative to galactic center
    R_sgr = np.sqrt(satellite_orbit.r[:,0]**2 + \
                    satellite_orbit.r[:,1]**2 + \
                    satellite_orbit.r[:,2]**2) * u.Unit(str(satellite_orbit.r.unit))
    
    v_halo = u.Quantity(potential["halo"]._unscaled_parameters["v_halo"].value,
                        str(potential["halo"]._unscaled_parameters["v_halo"].unit))
    m_disk = u.Quantity(potential["disk"]._unscaled_parameters["m"].value,
                        str(potential["disk"]._unscaled_parameters["m"].unit))
    m_bulge = u.Quantity(potential["bulge"]._unscaled_parameters["m"].value,
                        str(potential["bulge"]._unscaled_parameters["m"].unit))
                        
    m_halo_enc = v_halo**2 * R_sgr/G
    m_enc = m_disk + m_bulge + m_halo_enc
    
    # -----------------------------------------
    
    r_tide = R_sgr * ((m_sat / m_enc).decompose().value)**(1./3)
    v_esc = ( (G * m_sat / r_tide).decompose(bases=unit_bases)) ** 0.5
    
    R,V = relative_normalized_coordinates(particle_orbits, satellite_orbit, 
                                          r_tide, v_esc)
    D_ps = phase_space_distance(R,V)
    
    # Find the index of the time of the minimum D_ps for each particle
    min_time_idx = D_ps.argmin(axis=0)
    
    min_ps = np.zeros((Nparticles,6))
    for jj,ii in zip(min_time_idx, range(Nparticles)):
        min_ps[ii] = np.append(R[jj,ii], V[jj,ii])
    
    return min_ps
    
def generalized_variance(potential, particle_orbits, satellite_orbit):
    """ Compute the variance scalar that we will minimize.
        
        Parameters
        ----------
        potential : Potential
            The full Milky Way potential object.
        particle_orbits : TestParticleOrbit
            An object containing orbit information for a collection of 
            particles.
        satellite_orbit : TestParticleOrbit
            Data for the Sgr satellite center, interpolated onto the
            time grid for our particles.
    """
    
    min_ps = minimum_distance_matrix(potential, particle_orbits, satellite_orbit)
    
    cov_matrix = np.cov(min_ps.T)
    # cov_matrix -> (6 x 6) covariance matrix for particles
    w,v = np.linalg.eig(cov_matrix)
    
    return np.sum(w)

