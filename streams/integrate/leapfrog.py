# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy.utils import isiterable

__all__ = ["leapfrog"]

def leapfrog(acceleration_function, initial_position, initial_velocity, 
             t=None, t1=None, t2=None, dt=None, args=()):
             
    """ Given an acceleration function and initial conditions, integrate from 
        t1 to t2 with a timestep dt using Leapfrog integration. Alternatively,
        specify the full time array with 't'. The integration always *includes* 
        the final timestep.
        
        'acceleration_function' should accept a single parameter -- an array of
        position(s). The array should have shape (Npositions,Ndimensions), e.g.,
        for 100 particles in XYZ, the position array should be shape (100,3).
        
        For details on the algorithm, see: 
            http://ursa.as.arizona.edu/~rad/phys305/ODE_III/node11.html

        Parameters
        ----------
        acceleration_function : function
            A function that accepts a position or an array of positions and computes
            the acceleration at that position.
        initial_position : array, list
            A list or array of initial positions.
        initial_velocity : array, list
            A list or array of initial velocities.
    """
    
    initial_position = np.array(initial_position)
    initial_velocity = np.array(initial_velocity)
    
    if initial_position.shape != initial_velocity.shape:
        raise ValueError("initial_position shape must match initial_velocity "
                         "shape! {0} != {1}"
                         .format(initial_position.shape, 
                                 initial_velocity.shape))

    if initial_position.ndim == 1:
        # r_i just stands for positions, it's actually a vector
        r_im1 = np.array(initial_position).reshape(1, len(initial_position))
        v_im1 = np.array(initial_velocity).reshape(1, len(initial_position))
    else:
        r_im1 = initial_position
        v_im1 = initial_velocity
    
    if t is None and t1 is not None and t2 is not None and dt is not None:
        if not isiterable(dt):
            times = np.arange(t1, t2+dt, dt)
        else:
            ii = 0
            tt = t1
            times = []
            while tt < t2:
                times.append(tt)
                tt += dt[ii]
                ii += 1
            else:
                tt += dt[ii]
                times.append(tt)
            times = np.array(times)
            
    elif t is not None:
        times = t
    else:
        raise ValueError("Either specify t, or (t1, t2, and dt).")
    
    Ntimesteps = len(times)
    dt = times[1] - times[0]
    half_dt = 0.5*dt
    
    # Shape of final objects should be (Ntimesteps, Nparticles, Ndim)
    rs = np.zeros((Ntimesteps,) + r_im1.shape, dtype=np.float64)
    vs = np.zeros((Ntimesteps,) + v_im1.shape, dtype=np.float64)
    
    a_i = acceleration_function(r_im1, *args)
    v_im1_2 = v_im1 + a_i*half_dt
    v_i = v_im1_2 + a_i*half_dt
    r_i = r_im1
    for ii in range(Ntimesteps):
        rs[ii,:,:] = r_i
        vs[ii,:,:] = v_i
        
        r_i = r_im1 + v_im1_2*dt
        a_i = acceleration_function(r_i, *args)
        v_i = v_im1_2 + a_i*0.5*dt
        try:
            dt = times[ii+1]-times[ii]
        except IndexError:
            break
        v_ip1_2 = v_i + a_i*0.5*dt

        r_im1 = r_i
        v_im1_2 = v_ip1_2

    return times, rs, vs

def _tidal_radius(potential, r):
    """ Compute the tidal radius at a position r in the given potential.
        
        Parameters
        ----------
        potential : streams.CartesianPotential
    """
    _G = 4.4997533243534949e-12 # kpc^3 / Myr^2 / M_sun
    m_sat = 2.5E8 # u.M_sun
    R_orbit = np.sqrt(np.sum(r**2., axis=-1)) 
    
    m_halo_enc = potential["halo"]._parameters["v_halo"]**2 * R_orbit/_G
    m_enc = potential["disk"]._parameters["m"] + \
            potential["bulge"]._parameters["m"] + \
            m_halo_enc
    
    return R_orbit * (m_sat / m_enc)**(1./3)

def _adaptive_leapfrog(potential, initial_position, initial_velocity, 
                      t1, t2, args=(), resolution=3.):
    """ Given a potential and initial conditions, integrate from t1 to t2 
        with an adaptive timestep using Leapfrog integration. The timestep is
        chosen by computing the tidal radius of the satellite and taking:
            
            dt = (tidal radius) / (100 km/s) / resolution
        
        For details on the algorithm, see: 
            http://ursa.as.arizona.edu/~rad/phys305/ODE_III/node11.html

        Parameters
        ----------
        potential : CartesianPotential
            Represents the potential to integrate in.
        initial_position : array, list
            A list or array of initial positions.
        initial_velocity : array, list
            A list or array of initial velocities.
    """
    
    acceleration_function = potential._acceleration_at
    initial_position = np.array(initial_position)
    initial_velocity = np.array(initial_velocity)
    
    if initial_position.shape != initial_velocity.shape:
        raise ValueError("initial_position shape must match initial_velocity "
                         "shape! {0} != {1}"
                         .format(initial_position.shape, 
                                 initial_velocity.shape))

    if initial_position.ndim == 1:
        # r_i just stands for positions, it's actually a vector
        r_im1 = np.array(initial_position).reshape(1, len(initial_position))
        v_im1 = np.array(initial_velocity).reshape(1, len(initial_position))
    else:
        r_im1 = initial_position
        v_im1 = initial_velocity
    
    if t1 > t2:
        fac = -1.
    else:
        fac = 1.
    
    this_dt = _tidal_radius(potential, r_im1) / 0.102271 / resolution # kpc/Myr
    half_dt = 0.5*fac*this_dt
    
    # Shape of final objects should be (Ntimesteps, Nparticles, Ndim)
    rs = []
    vs = []
    
    a_i = acceleration_function(r_im1, *args)
    v_im1_2 = v_im1 + a_i*half_dt
    v_i = v_im1_2 + a_i*half_dt
    r_i = r_im1
    
    t = t1
    ii = 0
    times = []
    while t < fac*t2:
        rs.append(r_i)
        vs.append(v_i)
        
        r_i = r_im1 + v_im1_2*this_dt
        a_i = acceleration_function(r_i, *args)
        v_i = v_im1_2 + a_i*this_dt*0.5
        
        this_dt = fac*_tidal_radius(potential, r_i) / 0.102271 / resolution
        v_ip1_2 = v_i + a_i*this_dt*0.5

        r_im1 = r_i
        v_im1_2 = v_ip1_2
        
        ii += 1
        times.append(fac*t)
        t += fac*this_dt

    return np.array(times), np.array(rs), np.array(vs)
