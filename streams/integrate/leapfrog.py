# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy.utils import isiterable

from .timespec import _parse_time_specification

__all__ = ["LeapfrogIntegrator"]

class LeapfrogIntegrator(object):

    def __init__(self, acceleration_func, r_initial, v_initial, args=()):
        """ Initialize a leapfrog integrator given a function to compute
            the accelerations and initial conditions.

            Naming convention for variables:
                im1 -> i-1
                im1_2 -> i-1/2
                ip1 -> i+1
                ip1_2 -> i+1/2

            Initial position and velocity should have shape (ndim,nparticles)
            e.g., for 100 particles in 3D cartesian coordinates, the position
            array should have shape (100,3). For a single particle, (3,) is
            accepted and converted to (1,3).

            `acceleration_function` should accept the array of position(s) and
            optionally a set of arguments specified by `acceleration_args`.

            For details on the algorithm, see:
                http://ursa.as.arizona.edu/~rad/phys305/ODE_III/node11.html

            Parameters
            ----------
            acceleration_func : func
            r_initial : array_like
            v_initial : array_like
            args : tuple (optional)
                Any extra arguments for the acceleration function.

        """

        if not hasattr(acceleration_func, '__call__'):
            raise ValueError("acceleration_func must be a callable object, "
                        "e.g. a function, that evaluates the acceleration "
                        "at a given position")

        self.acc = acceleration_func
        self._acc_args = args

        # Validate initial conditions
        if not isinstance(r_initial, np.ndarray):
            r_initial = np.array(r_initial)

        if not isinstance(v_initial, np.ndarray):
            v_initial = np.array(v_initial)

        if not r_initial.shape == v_initial.shape:
            raise ValueError("Shape of positions must match velocities")
        elif r_initial.ndim == 1:
            r_initial = r_initial[np.newaxis]
            v_initial = v_initial[np.newaxis]
        elif r_initial.ndim > 2:
            raise ValueError("Initial conditions should have shape "
                             "(nparticles,ndim) or (ndim,).")

        self.r_im1 = r_initial
        self.v_im1 = v_initial
        self.v_im1_2 = None
        self._dt = None

    def _position_step(self, r, v, dt):
        """ The 'drift' part of the leapfrog integration. Update the positions
            given a velocity.
        """
        return r + v*dt

    def _velocity_halfstep(self, a, v, dt):
        """ The 'kick' part of the leapfrog integration. Update the velocities
            given a velocity.
        """
        return v + a*self._half_dt

    def step(self, dt):
        """ Step forward the positions and velocities by the given timestep """

        if self._dt is None:
            self._dt = dt

        r_i = self._position_step(self.r_im1, self.v_im1_2, self._dt)
        a_i = self.acc(r_i, *self._acc_args)
        v_i = self._velocity_halfstep(a_i, self.v_im1_2, self._dt)

        self._dt = dt
        v_ip1_2 = self._velocity_halfstep(a_i, v_i, self._dt)

        self.r_im1 = r_i
        self.v_im1 = v_i
        self.v_im1_2 = v_ip1_2

        return r_i, v_i

    def _prime(self, dt):
        """ Leapfrog updates the velocities offset a half-step from the
            position updates. If we're given initial conditions aligned in
            time, e.g. the positions and velocities at the same 0th step,
            then we have to initially scoot the velocities forward by a half
            step to prime the integrator.
        """

        self._dt = dt
        self._half_dt = dt/2.

        # If the integrator has not been manually primed or run previously,
        #   here is where we scoot the velocity at time=0 to v(t+1/2)
        if self.v_im1_2 is None:
            a_im1 = self.acc(self.r_im1, *self._acc_args)
            self.v_im1_2 = self._velocity_halfstep(a_im1,
                                                   self.v_im1,
                                                   dt)

    def run(self, **time_spec):
        """ Run the integrator given a time specification. There are a few
            combinations of kwargs that are accepted -- see below.

            Parameters (kwargs)
            -------------------
            dt, Nsteps[, t1] : (numeric, int[, numeric])
                A fixed timestep dt and a number of steps to run for.
            dt, t1, t2 : (numeric, numeric, numeric)
                A fixed timestep dt, an initial time, and a final time.
            dt, t1 : (array_like, numeric)
                An array of timesteps dt and an initial time.
            t : array_like
                An array of times (dts = t[1:] - t[:-1])

        """

        times = _parse_time_specification(**time_spec)
        dts = times[1:]-times[:-1]
        Ntimesteps = len(times)

        self._prime(dts[0])

        rs = np.zeros((Ntimesteps,) + self.r_im1.shape, dtype=float)
        vs = np.zeros((Ntimesteps,) + self.v_im1.shape, dtype=float)

        # Set first step to the initial conditions
        rs[0] = self.r_im1
        vs[0] = self.v_im1

        for ii,dt in enumerate(dts):
            r_i, v_i = self.step(dt)
            rs[ii+1] = r_i
            vs[ii+1] = v_i

        return times, rs, vs