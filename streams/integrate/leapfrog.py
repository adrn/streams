# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

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
    
    if t == None:           
        # t2+dt to include the last data point
        times = np.arange(t1, t2+dt, dt)
    else:
        times = t
        dt = times[1]-times[0]
    
    Ntimesteps = len(times)
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
        v_i = v_im1_2 + a_i*half_dt
        v_ip1_2 = v_i + a_i*half_dt

        r_im1 = r_i
        v_im1_2 = v_ip1_2

    return times, rs, vs

