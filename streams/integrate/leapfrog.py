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

def _parse_time_specification(dt=None, Nsteps=None, t1=None, t2=None, t=None):
    """ Return a list of times given a few combinations of kwargs that are 
        accepted -- see below.
            
        Parameters
        ----------
        dt, Nsteps[, t1] : (numeric, int[, numeric])
            A fixed timestep dt and a number of steps to run for.
        dt, t1, t2 : (numeric, numeric[, numeric])
            A fixed timestep dt, an initial time, and an final time.
        Nsteps, t1, t2 : (int, numeric, numeric)
            Number of steps between an initial time, and a final time.
        t : array_like
            An array of times (dts = t[1:] - t[:-1])
        
        """
    # t : array_like
    if t is not None:
        times = t
        return times
        
    else:
        if dt is None and t1 is None and t2 is None:
            raise ValueError("If a full array of times is not specified"
                             " you must specify a timestep.")
        
        # dt, t1 : (array_like, numeric)
        elif isinstance(dt, np.ndarray):
            if t1 is None or t2 is not None:
                raise ValueError("If dt is given as an array, you must "
                                 "specify the starting time t1 and *not* "
                                 "the end time, t2.")
            
            times = np.cumsum(np.append([0.], dt)) + t1
            return times
            
        else:            
            # dt, Nsteps : (numeric, int)
            if Nsteps is not None:
                if t1 is None:
                    t1 = 0.
                
                if t2 is None:
                    return _parse_time_specification(dt=np.ones(Nsteps)*dt, 
                                                     t1=t1)
                
                # Nsteps, t1, t2 : (int, numeric, numeric)
                else:
                    return np.linspace(t1, t2, Nsteps, endpoint=True)
            
            elif t2 is not None:
                if t1 is None:
                    t1 = 0.
                
                ii = 0
                times = [t1]
                while (times[-1] < t2) and (ii < 1E6):
                    times.append(times[-1]+dt)
                    ii += 1
                
                return np.array(times)

class LeapfrogIntegrator(object):
    
    def __init__(self, acceleration_func, r_initial, v_initial, acceleration_args):
        """ Initialize a leapfrog integrator given a function to compute
            the accelerations and initial conditions. 
        
            Naming convention for variables:
                im1 -> i-1
                im1_2 -> i-1/2
                ip1 -> i+1
                ip1_2 -> i+1/2
            
            Initial position and velocity should have shape (Nparticles, Ndim),
            so even 
            
            Parameters
            ----------
            acceleration_func : func
            r_initial : array_like
            v_initial : array_like
            acceleration_args : tuple
                Any extra arguments for the acceleration function.
            
        """
        
        if not hasattr(acceleration_func, '__call__'):
            raise ValueError("acceleration_func must be a callable object, "
                             "e.g. a function, that evaluates the acceleration "
                             "at a given position")
        
        self.acc = acceleration_func
        self._acc_args = acceleration_args
        
        # Validate initial conditions
        if not isinstance(r_initial, np.ndarray):
            r_initial = np.array(r_initial)
        
        if not isinstance(v_initial, np.ndarray):
            v_initial = np.array(v_initial)
        
        if not r_initial.shape == v_initial.shape:
            raise ValueError("Shape of positions must match velocities")
        elif r_initial.ndim != 2:
            raise ValueError("Initial conditions should have shape "
                             "(n_particles, n_dim).")
            
        self.r_im1 = r_initial
        self.v_im1 = v_initial
        self.v_im1_2 = None
        self.dt = None
        
    def _position_step(self, r, v, dt):
        """ The 'drift' part of the leapfrog integration. Update the positions
            given a velocity.
        """
        return r + v*dt
        
    def _velocity_halfstep(self, r, v, dt):
        """ The 'kick' part of the leapfrog integration. Update the velocities
            given a velocity.
        """
        half_dt = 0.5*dt
        a_i = self.acc(r, *self._acc_args)
        return v + a_i*half_dt
        
    def step(self, dt):
        """ Step forward the positions and velocities by the given timestep """
        
        if self._dt is None:
            self._dt = dt
        
        r_i = self._position_step(self.r_im1, self.v_im1_2, self._dt)
        v_i = self._velocity_halfstep(r_i, self.v_im1_2, self._dt)
        
        self._dt = dt
        v_ip1_2 = self._velocity_halfstep(r_i, v_i, self._dt)
        
        self.r_im1 = r_i
        self.v_im1_2 = v_ip1_2
        
        return r_i, v_i
    
    def _prime(self, dt):
        """ Leapfrog updates the velocities offset a half-step from the 
            position updates. If we're given initial conditions aligned in
            time, e.g. the positions and velocities at the same 0th step, 
            then we have to initiall scoot the velocities forward by a half 
            step to prime the integrator.
        """
            
        self._dt = dt
        
        # If the integrator has not been manually primed or run previously,
        #   here is where we scoot the velocity at time=0 to v(t+1/2)
        if self.v_im1_2 is None:
            self.v_im1_2 = self._velocity_halfstep(self.r_im1, 
                                                   self.v_im1,
                                                   dt)
    
    def run(self, dt=None, Nsteps=None, t1=None, t2=None, t=None):
        """ Run the integrator given a time specification. There are a few
            combinations of kwargs that are accepted -- see below.
            
            Parameters
            ----------
            dt, Nsteps[, t1] : (numeric, int[, numeric])
                A fixed timestep dt and a number of steps to run for.
            dt, t1, t2 : (numeric, numeric, numeric)
                A fixed timestep dt, an initial time, and a final time.
            dt, t1 : (array_like, numeric)
                An array of timesteps dt and an initial time.
            t : array_like
                An array of times (dts = t[1:] - t[:-1])
            
        """
        
        times = _parse_time_specification(dt=dt, Nsteps=Nsteps, 
                                          t1=t1, t2=t2, t=t)
        dts = times[1:]-times[:-1]
        
        # HERE
        self._prime(dts[0])
        
        rs,vs = [],[]
        rs.append(self.r_im1)
        vs.append(self.v_im1)
        for dt in dts:
            r_i, v_i = self.step(dt)
            rs.append(r_i)
            vs.append(v_i)
        
        return np.array(rs), np.array(vs)

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
