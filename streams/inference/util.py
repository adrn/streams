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

__all__ = ["minimum_distance_matrix",
           "phase_space_distance",
           "relative_normalized_coordinates",
           "generalized_variance"]

def relative_normalized_coordinates(potential,
                                    satellite_orbit,
                                    particle_orbits):
    """ Compute the coordinates of particles relative to the satellite,
        with positions normalized by the tidal radius and velocities
        normalized by the escape velocity.

        Note::
            Assumes the particle and satellite position/velocity
            units are the same!

        Parameters
        ----------
        potential : streams.CartesianPotential
        satellite_orbit : Orbit
        particle_orbits : Orbit
    """

    # need to add a new axis to normalize each coordinate component
    r_tide = potential._tidal_radius(m=satellite_orbit._m,
                                     r=satellite_orbit._r)[:,:,np.newaxis]

    v_esc = potential._escape_velocity(m=satellite_orbit._m,
                                       r_tide=r_tide)

    return (particle_orbits._r - satellite_orbit._r) / r_tide, \
           (particle_orbits._v - satellite_orbit._v) / v_esc

def phase_space_distance(potential, satellite_orbit, particle_orbits):
    """ Compute the phase-space distance for a set of particles relative
        to a satellite in a given potential.

        Parameters
        ----------
        potential : streams.CartesianPotential
        satellite_orbit : Orbit
        particle_orbits : Orbit
    """
    return np.sqrt(np.sum(R**2, axis=-1) + np.sum(V**2, axis=-1))

def minimum_distance_matrix(potential, satellite_orbit, particle_orbits):
    """ Compute the Nx6 matrix of minimum phase-space distance vectors.

        Parameters
        ----------
        potential : Potential
        satellite_orbit : Orbit
        particle_orbits : Orbit
    """

    R,V = relative_normalized_coordinates(potential,
                                          satellite_orbit, particle_orbits)
    D_ps = np.sqrt(np.sum(R**2, axis=-1) + np.sum(V**2, axis=-1))

    # Find the index of the time of the minimum D_ps for each particle
    min_time_idx = D_ps.argmin(axis=0)
    min_ps = np.zeros((particle_orbits.nparticles,6))

    xx = zip(min_time_idx, range(particle_orbits.nparticles))
    for kk in range(particle_orbits.nparticles):
        jj,ii = xx[kk]
        min_ps[ii] = np.append(R[jj,ii], V[jj,ii])

    return min_ps

def generalized_variance(potential, satellite_orbit, particle_orbits):
    """ Compute the variance scalar -- variance of the minimum phase-
        space distance matrix.

        Parameters
        ----------
        potential : Potential
            The full Milky Way potential object.
        particle_orbits : Orbit
            An object containing orbit information for a collection of
            particles.
        satellite_orbit : Orbit
            Data for the Sgr satellite center, interpolated onto the
            time grid for our particles.
    """

    min_ps = minimum_distance_matrix(potential, satellite_orbit, particle_orbits)
    cov_matrix = np.cov(np.fabs(min_ps.T))

    sign,logdet = np.linalg.slogdet(cov_matrix)
    return logdet
