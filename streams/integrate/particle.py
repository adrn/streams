# coding: utf-8

""" Wrapper around LeapfrogIntegrator that knows how to handle
    Particle objects.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

# Project
from .leapfrog import LeapfrogIntegrator
from ..dynamics import Particle

# Create logger
logger = logging.getLogger(__name__)

class ParticleIntegrator(LeapfrogIntegrator):

    def __init__(self, particles, potential, args=()):
        """ TODO: describe...

            Parameters
            ----------
            particles : Particle or iterable
                A Particle, or list/tuple of Particle objects.
        """

        self.particles = particles
        super(ParticleIntegrator,self).__init__(potential._acceleration_at,
                                                self.X0[:3], self.X0[3:],
                                                args=args)

    @property
    def X0(self):
        # Stack positions and velocities from particles
        particles = self.particles
        if isinstance(particles, Particle):
            X0 = particles._X
        else:
            X0 = np.hstack([p._X for p in particles])

        return X0

    def run(self, **time_spec):
        self.r_im1 = self.X0[:3]
        self.v_im1 = self.X0[3:]

        t,r,v = super(ParticleIntegrator,self).run(**time_spec)

        return t,r,v