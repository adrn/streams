# coding: utf-8

""" Helper function for turning different ways of specifying the integration
    times into an array of times.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

def _parse_time_specification(dt=None, Nsteps=None, t1=None, t2=None, t=None):
    """ Return an array of times given a few combinations of kwargs that are 
        accepted -- see below.
            
        Parameters
        ----------
        dt, Nsteps[, t1] : (numeric, int[, numeric])
            A fixed timestep dt and a number of steps to run for.
        dt, t1, t2 : (numeric, numeric, numeric)
            A fixed timestep dt, an initial time, and an final time.
        dt, t1 : (array_like, numeric)
            An array of timesteps dt and an initial time.
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
        if dt is None and (t1 is None or t2 is None or Nsteps is None):
            raise ValueError("Invalid spec. See docstring.")
        
        # dt, Nsteps[, t1] : (numeric, int[, numeric])
        elif dt is not None and Nsteps is not None:
            if t1 is None:
                t1 = 0.
            
            times = _parse_time_specification(dt=np.ones(Nsteps)*dt, 
                                              t1=t1)
        # dt, t1, t2 : (numeric, numeric, numeric)
        elif dt is not None and t1 is not None and t2 is not None:            
            if t2 < t1 and dt < 0.:
                fac = -1.
            elif t2 > t1 and dt > 0.:
                fac = 1.
            else:
                raise ValueError("If t2 < t1, dt must be negative. If t1 < t2, "
                                 "dt should be positive.")
            
            ii = 0
            next_t = t1
            times = [next_t]
            while (fac*next_t < fac*t2) and (ii < 1E6):
                times.append(next_t)
                ii += 1
                next_t = times[-1]+dt
            
            if times[-1] != t2:
                times.append(t2)
            
            times = np.array(times)
    
        # dt, t1 : (array_like, numeric)
        elif isinstance(dt, np.ndarray) and t1 is not None:            
            times = np.cumsum(np.append([0.], dt)) + t1
            
        # Nsteps, t1, t2 : (int, numeric, numeric)
        elif dt is None and not (t1 is None or t2 is None or Nsteps is None):
            times = np.linspace(t1, t2, Nsteps, endpoint=True)
        
        else:
            raise ValueError("Invalid options. See docstring.")
        
        return times