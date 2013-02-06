# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable

from ..potential import Potential
from ..integrate import leapfrog

__all__ = ["Particle", "ParticleSimulation"]

class ParticleSimulation(object):

    def __init__(self, potential):
        """ This is a handy class that will handle integrating a group of particles through a potential. """

        if not isinstance(potential, Potential):
            raise TypeError("potential must be a streams.Potential object.")

        self.potential = potential
        #self.particles = dict()
        self.particles = list()
        self._particle_map = dict()

        self._particle_pos_array = None
        self._particle_vel_array = None

    def add_particle(self, p):
        """ Add a test-particle to the simulation (to be integrated through the potential) """

        if not isinstance(p, Particle):
            raise TypeError("add_particle() only accepts Particle objects.")

        self.particles.append(p)

        if self._particle_pos_array == None:
            self._particle_pos_array = p.position.reshape((1,p.position.shape[0]))
            self._particle_vel_array = p.velocity.reshape((1,p.velocity.shape[0]))
        else:
            self._particle_pos_array = np.append(self._particle_pos_array, p.position.reshape((1,p.position.shape[0])), axis=0)
            self._particle_vel_array = np.append(self._particle_vel_array, p.velocity.reshape((1,p.velocity.shape[0])), axis=0)

        #self._particle_map[hash(p)] = len(self._particle_pos_array)-1

    def run(self, t1, t2, dt=None, integrator=leapfrog):
        """ Integrate the particle positions from t1 to t2 using the specified integrator """

        if dt == None:
            dt = (t2 - t1) / 100.
        ts, xs, vs = integrator(self.potential.acceleration_at, self._particle_pos_array, self._particle_vel_array, t1, t2, dt)

        return ts, xs, vs

    def particles_at(self, t):
        """ Get a list of Particle objects at the given time step """


class Particle(object):

    def __init__(self, position, velocity, mass=1.0):
        """ Represents a massive particle at some position with some velocity """

        self.marker = "."
        self.color = "k"
        self.alpha = 0.4

        # Set position as array
        if not isiterable(position):
            position = [position]

        self.position = np.array(position, copy=True)

        # Set velocity as array
        if not isiterable(velocity):
            velocity = [velocity]

        self.velocity = np.array(velocity, copy=True)

        self.mass = mass

        if self.position.shape != self.velocity.shape:
            raise ValueError("position and velocity must have the same shape!")

        if self.position.ndim > 1:
            raise TypeError("Position/velocity arrays may only be one dimensional, e.g. may only represent a single particle.")

    def __repr__(self):
        return "<Particle M={0} at x=[{1}], v=[{2}]".format(str(self.mass), ",".join(map(str, self.position)), ",".join(map(str, self.velocity)))

    def __key(self):
        return (tuple(self.position), tuple(self.velocity), self.mass)

    def __hash__(self):
        return hash(self.__key())