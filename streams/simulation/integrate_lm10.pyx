"""
Deimos:
cython -a _common.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -o _common.so _common.c

Laptop:
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I /usr/include/python2.7 -L /usr/lib/python2.7 -l python -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -o _common.so _common.c
"""
from __future__ import division

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double fmod(double, double)
    double floor(double)
    double fmax(double, double)
    double fmin(double, double)
    double sqrt(double)
    double log(double)
    double sin(double)
    double cos(double)
    int isnan(double)
    double fabs(double)

@cython.boundscheck(False) # turn of bounds-checking for entire function
def lm10_acceleration(np.ndarray[double, ndim=2] r, long Nparticles, 
                      double q1, double qz, double phi, double v_halo):
    
    data = np.zeros((Nparticles,3), dtype=np.float64)
    cdef double _tmp, x, y, z, xx, yy, zz, fac, R
    
    cdef double G, a, b, c, m_disk, m_bulge
    G = 4.49975332435e-12 # kpc^3 / Myr^2 / M_sun
    
    # Miyamoto-Nagai
    a = 6.5 # kpc
    b = 0.26 # kpc
    m_disk = 1.E11 # M_sun
    
    # Hernquist
    c = 0.7 # kpc
    m_bulge = 3.4E10 # M_sun
    
    # Halo
    cdef double q1_sq, q2_sq, qz_sq, r_halo_sq, v_halo_sq
    q1_sq = q1*q1
    q2_sq = 1.
    qz_sq = qz*qz
    r_halo_sq = 12.*12 # kpc
    v_halo_sq = v_halo*v_halo
    
    for ii in range(Nparticles):
        x = r[ii][0]
        y = r[ii][1]
        z = r[ii][2]
        
        xx = x*x
        yy = y*y
        zz = z*z
        
        # Disk
        fac1 = G*m_disk*((xx + yy) + (a + sqrt(zz + b**2))**2)**-1.5
        _tmp = a/(sqrt(zz + b**2))
        
        # Bulge
        R = sqrt(xx + yy + zz)
        fac2 = G*m_bulge / ((R + c)**2 * R)
        
        # Halo
        C1 = (cos(phi)/q1)**2+(sin(phi)/q2)**2
        C2 = (cos(phi)/q2)**2+(sin(phi)/q1)**2
        C3 = 2.*sin(phi)*cos(phi)*(1./q1_sq - 1./q2_sq)
        
        fac3 = v_halo_sq / (C1*xx + C2*yy + C3*x*y + zz/qz_sq + r_halo_sq)
        
        data[ii][0] = fac1*x + fac2*x + fac3*(2.*C1*x + C3*y)
        data[ii][1] = fac1*yy + fac2*yy + fac3*(2.*C2*y + C3*x)
        data[ii][2] = fac1*zz * (1.+_tmp) + fac2*zz + 2.*fac3*z/qz_sq
            
    return data

'''

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
    
    if initial_position.shape != initial_velocity.shape:
        raise ValueError("initial_position shape must match initial_velocity "
                         "shape! {0} != {1}"
                         .format(initial_position.shape, 
                                 initial_velocity.shape))

    if initial_position.ndim == 1:
        # r_i just stands for positions, it's actually a vector
        r_i = np.array(initial_position)\
                .reshape(1, len(initial_position))
        v_i = np.array(initial_velocity)\
                .reshape(1, len(initial_position))
    else:
        r_i = initial_position
        v_i = initial_velocity
    
    if t == None:           
        times = np.arange(t1, t2+dt, dt)
        #times = np.arange(t1, t2, dt)
    else:
        times = t
        dt = times[1]-times[0]
    
    Ntimesteps = len(times)

    # Shape of final object should be (Ntimesteps, Ndim, Nparticles)
    rs = np.zeros((Ntimesteps,) + r_i.shape, dtype=np.float64)
    vs = np.zeros((Ntimesteps,) + v_i.shape, dtype=np.float64)

    for ii in range(Ntimesteps):
        t = times[ii]
        a_i = acceleration_function(r_i).T
        
        r_ip1 = r_i + v_i*dt + 0.5*a_i*dt*dt        
        a_ip1 = acceleration_function(r_ip1).T
        v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt

        rs[ii,:,:] = r_i
        vs[ii,:,:] = v_i

        a_i = a_ip1
        r_i = r_ip1
        v_i = v_ip1

    return times, rs, vs
'''