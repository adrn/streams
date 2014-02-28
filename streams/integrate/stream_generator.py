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

__all__ = ["StreamGenerator"]

class StreamGenerator(object):

    def __init__(self, potential, satellite_orbit, satellite_mass, acc_args=()):
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

        if not hasattr(potential, '_acceleration_at'):
            raise ValueError("Potential must have an acceleration method.")

        self.potential = potential
        self._acc_args = acc_args

        # Validate initial conditions
        if not isinstance(satellite_orbit, np.ndarray):
            satellite_orbit = np.array(satellite_orbit)

        self.satellite_orbit = satellite_orbit
        self.satellite_mass = satellite_mass

    def step(self, dt):
        """ Step forward the positions and velocities by the given timestep """

        self.r_im1 += self.v_im1_2*dt
        a_i = self.acc(self.r_im1, *self._acc_args)

        self.v_im1 += a_i*dt
        self.v_im1_2 += a_i*dt

        return self.r_im1, self.v_im1

    def run(self, tub_idxs, t1, t2, dt):
        """ Run the integrator given a time specification. There are a few
            combinations of kwargs that are accepted -- see below.

            Parameters
            ----------
            t1, t2, dt : (numeric, numeric, numeric)
                A fixed timestep dt, an initial time, and a final time.

        """
        tub_idxs = np.array(tub_idxs)

        if dt < 0.:
            raise ValueError("Only forward integration supported (dt>0).")

        times = _parse_time_specification(t1=t1, t2=t2, dt=dt)
        ntimesteps = len(times)
        nparticles = len(tub_idxs)

        # make a container for the particle orbits
        orbits = np.zeros((ntimesteps, nparticles, 6), dtype=float)

        v_im1_2 = np.zeros((nparticles,3))
        v_im1 = np.zeros((nparticles,3))
        r_im1 = np.zeros((nparticles,3))
        a_im1 = np.zeros((nparticles,3))

        release_idx = tub_idxs

        ###
        # compute properties of the disruption
        satellite_R = np.sqrt(np.sum(self.satellite_orbit[...,:3]**2, axis=-1))
        r_tide = self.potential._tidal_radius(self.satellite_mass, self.satellite_orbit)#*1.4

        f = r_tide / satellite_R
        v_disp = np.sqrt(np.sum(self.satellite_orbit[...,3:]**2, axis=-1)) * f #/ 1.4

        r_tide /= 4.
        v_disp /= 2.5

        #############
        init_r = np.zeros((nparticles,3))
        init_v = np.zeros((nparticles,3))
        tail_bit = np.random.uniform(-1., 1., size=nparticles)
        tail_bit = np.sign(tail_bit)
        for ii,idx in enumerate(release_idx):
            a_pm = np.squeeze((satellite_R[idx] + r_tide[idx]*tail_bit[ii]) / satellite_R[idx])
            r = self.satellite_orbit[idx,0,:3]
            v = self.satellite_orbit[idx,0,3:]
            init_r[ii] = np.random.normal(r + a_pm*r_tide[idx], r_tide[idx])
            init_v[ii] = np.random.normal(v, v_disp[idx])

        #############

        acc = self.potential._acceleration_at
        for ii,t in enumerate(times):
            if np.any(release_idx == ii):
                # half step forward those velocities
                _npart = sum(release_idx == ii)
                _acc = np.zeros((_npart,3))
                r_im1[release_idx == ii] = init_r[release_idx == ii]
                v_im1[release_idx == ii] = init_v[release_idx == ii]

                a_im1[release_idx == ii] = acc(r_im1[release_idx == ii], _npart, _acc)
                v_im1_2[release_idx == ii] = init_v[release_idx == ii] + \
                                                a_im1[release_idx == ii]*dt/2.

            orbits[ii,:,:3] = r_im1
            orbits[ii,:,3:] = v_im1

            r_im1 = r_im1 + v_im1_2*dt
            v_im1 = v_im1_2 + a_im1*dt/2.

            a_im1 = acc(r_im1, *self._acc_args)
            v_im1_2 = v_im1_2 + a_im1*dt

        return times, orbits