# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["leapfrog"]

def leapfrog(acceleration_function, initial_position, initial_velocity, t=None, 
             t1=None, t2=None, dt=None):
             
    """ Given an acceleration function and initial conditions, integrate from 
        t1 to t2 with a timestep dt using Leapfrog integration. Alternatively,
        specify the full time array with 't'. The integration always *includes* 
        the final timestep!
        See: http://ursa.as.arizona.edu/~rad/phys305/ODE_III/node11.html

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
    
    if not isinstance(initial_position, u.Quantity) or \
        not isinstance(initial_velocity, u.Quantity):
        raise TypeError("Initial position and velocities must be Quantity "
                        "objects, not {0},{1}"
                        .format(type(initial_position),type(initial_velocity)))

    if initial_position.value.shape != initial_velocity.value.shape:
        raise ValueError("initial_position shape must match initial_velocity "
                         "shape! {0} != {1}"
                         .format(initial_position.value.shape, 
                                 initial_velocity.value.shape))

    if initial_position.value.ndim == 1:
        # r_i just stands for positions, it's actually a vector
        r_i = np.array(initial_position.value)\
                .reshape(1, len(initial_position))*initial_position.unit
        v_i = np.array(initial_velocity.value)\
                .reshape(1, len(initial_position))*initial_velocity.unit
    else:
        r_i = initial_position
        v_i = initial_velocity
    
    if t == None:
        if not isinstance(t1, u.Quantity) or not isinstance(t2, u.Quantity) \
            or not isinstance(dt, u.Quantity):
            raise TypeError("Times must be astropy Quantity objects")
            
        times = np.arange(t1.value, 
                          (t2.to(t1.unit)+dt.to(t1.unit)).value, 
                          dt.to(t1.unit).value) * t1.unit
        #times = np.arange(t1, t2, dt)
    else:
        if not isinstance(t, u.Quantity):
            raise TypeError("Times must be astropy Quantity objects")
        times = t        
        dt = times[1]-times[0]
    
    Ntimesteps = len(times)

    # Shape of final object should be (Ntimesteps, Ndim, Nparticles)
    rs = np.zeros((Ntimesteps,) + r_i.value.shape, dtype=np.float64)
    vs = np.zeros((Ntimesteps,) + v_i.value.shape, dtype=np.float64)

    for ii in range(Ntimesteps):
        t = times[ii]
        a_i = acceleration_function(r_i)
        a_i = a_i.value.T*a_i.unit
        
        r_ip1 = r_i + v_i*dt + 0.5*a_i*dt*dt        
        a_ip1 = acceleration_function(r_ip1)
        a_ip1 = a_ip1.value.T*a_ip1.unit
        v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt

        rs[ii,:,:] = r_i.value
        vs[ii,:,:] = v_i.value

        a_i = a_ip1
        r_i = r_ip1
        v_i = v_ip1

    return times, rs*r_i.unit, vs*v_i.unit

