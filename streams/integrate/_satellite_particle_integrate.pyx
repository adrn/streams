from __future__ import division

import sys
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True) 
@cython.wraparound(False)
@cython.nonecheck(False)
def _lm10_integrator(double[:, ::1] r not None, double[:, ::1] v not None,
                     acceleration, int n_particles, double resolution,
                     float t1, float t2,
                     timestep_func, timestep_args):
    """ r contains the satellite position + all particle positions.
        v contains the satellite velocity + all particle velocities.
    """
    
    n_particles += 1
    
    dt_i = dt_im1 = -0.1
    
    # create data array for acclerations
    #cdef double[:, ::1] a_i = np.empty((n_particles, 3))
    cdef np.ndarray[double, ndim=2, mode='c'] a_i = np.empty((n_particles, 3))
    
    # prime the integrator -- do a half step in v
    r_im1 = r
    a_im1 = acceleration(r_im1, n_particles=n_particles, a_i=a_i) # or whatever
    v_im1 = v
    v_im1_2 = v + a_im1*dt_im1
    
    # create empty arrays for positions and velocities
    Ntimesteps = int(10000.*resolution)
    cdef double[:, :, ::1] rs = np.empty((Ntimesteps, n_particles, 3))
    cdef double[:, :, ::1] vs = np.empty((Ntimesteps, n_particles, 3))
    rs[0] = r_im1
    vs[0] = v_im1
    
    # container for times
    times = [t1]
    this_t = t1
    for ii in range(Ntimesteps):
        dt = 0.5*(dt_im1 + dt_i)
        half_dt = dt*0.5
    
        #r_i v_i = step(dt)
        r_i = r_im1 + v_im1_2*dt
        a_i = acceleration(r_i, n_particles, a_i) # or whatever
        v_i = v_im1_2 + a_i*half_dt
        
        rs[ii] = r_i
        vs[ii] = v_i
        
        v_ip1_2 = v_i + a_i*half_dt 
        
        # HMM?
        dt_i = timestep_func(r_i, v_i, *timestep_args) / resolution
        this_t = this_t + dt
        times.append(this_t)
        dt_im1 = dt_i
        ii += 1
        
        r_im1 = r_i
        v_im1 = v_i
        v_im1_2 = v_ip1_2
        
        if this_t > t2:
            break