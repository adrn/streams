# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import uuid

# Third-party
import numpy as np
from ..potential import Potential
from ..util import _validate_coord

__all__ = ["PotentialIntegrator", "leapfrog"]

def leapfrog(acceleration_function, initial_position, initial_velocity, t1, t2, dt=None):
    ''' '''

    initial_position = np.array(initial_position)
    initial_velocity = np.array(initial_velocity)

    if initial_position.ndim == 1:
        # x_i just stands for positions, it's actually a vector
        x_i = np.array(initial_position).reshape(1, len(initial_position))
        v_i = np.array(initial_velocity).reshape(1, len(initial_position))
    else:
        x_i = np.array(initial_position)
        v_i = np.array(initial_velocity)

    times = np.arange(t1, t2+dt, dt)
    Ntimesteps = len(times)

    # Shape of final object should be (Ntimesteps, Nparticles, Ndim)
    xs = np.zeros((Ntimesteps,) + x_i.shape, dtype=np.float64)
    vs = np.zeros((Ntimesteps,) + v_i.shape, dtype=np.float64)

    for ii in range(Ntimesteps):
        t = times[ii]
        a_i = acceleration_function(x_i)

        x_ip1 = x_i + v_i*dt + 0.5*a_i*dt*dt
        a_ip1 = acceleration_function(x_ip1)
        v_ip1 = v_i + 0.5*(a_i + a_ip1)*dt

        xs[ii,:,:] = x_i
        vs[ii,:,:] = v_i

        a_i = a_ip1
        x_i = x_ip1
        v_i = v_ip1

    return times, xs, vs

class PotentialIntegrator(object):

    def __init__(self, potential, integrator=None):
        ''' Convenience class for integrating particles in a potential.

            Parameters
            ----------
            potential : Potential
            integrator : function (optional)
                The integration scheme. Defaults to leapfrog.

        '''
        if not isinstance(potential, Potential):
            raise TypeError("potential must be a Potential object or subclass.")