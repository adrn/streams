# cython -a _leapfrog.pyx; gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I /usr/include/python2.7 -L /usr/lib/python2.7 -l python -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -o _leapfrog.so _leapfrog.c
from __future__ import division

import sys
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double fmod(double, double)
    double floor(double)
    double fmax(double, double)
    double fmin(double, double)
    double sqrt(double)
    int isnan(double)
    double fabs(double)

def leapfrog2(acceleration_function, initial_position, initial_velocity, t1, t2, dt=None):
    ''' '''

    initial_position = np.array(initial_position)
    initial_velocity = np.array(initial_velocity)

    if initial_position.ndim == 1:
        # x_i just stands for positions, it's actually a vector
        x_i = np.array(initial_position).reshape(len(initial_position), 1)
        v_i = np.array(initial_velocity).reshape(len(initial_position), 1)
    else:
        x_i = np.array(initial_position)
        v_i = np.array(initial_velocity)

    times = np.arange(t1, t2+dt, dt)
    Ntimesteps = len(times)

    # Shape of final object should be (Ntimesteps, Nparticles, Ndim)
    xs = np.zeros((len(times),) + x_i.shape)
    vs = np.zeros((len(times),) + v_i.shape)

    for ii in range(Ntimesteps):
        t = times[ii]
        a_i = acceleration_function(x_i)
        x_ip1 = x_i + v_i*dt + 0.5*a_i*dt*dt
        a_ip1 = acceleration_function(x_i)
        v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt

        xs[ii,...] = x_i
        vs[ii,...] = v_i

        a_i = a_ip1
        x_i = x_ip1
        v_i = v_ip1

    return times, xs, vs