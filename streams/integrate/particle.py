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
from ..dynamics import Particle, Orbit

# Create logger
logger = logging.getLogger(__name__)

__all__ = ["ParticleIntegrator"]

class ParticleIntegrator(LeapfrogIntegrator):

    def __init__(self, particles, potential, args=()):
        """ TODO: describe...

            Parameters
            ----------
            particles : Particle or iterable
                A Particle, or list/tuple of Particle objects.
        """

        if isinstance(particles, Particle):
            self.X0 = particles._X
            particles = [particles]
        else:
            self.X0 = np.vstack([p._X for p in particles])
        self.particles = particles

        super(ParticleIntegrator,self).__init__(potential._acceleration_at,
                                            self.X0[...,:3], self.X0[...,3:],
                                            args=args)

    def run(self, **time_spec):
        self.r_im1 = np.array(self.X0[...,:3])
        self.v_im1 = np.array(self.X0[...,3:])

        t,r,v = super(ParticleIntegrator,self).run(**time_spec)
        t = t*u.Myr # HACK!!!

        orbits = []
        ix = 0
        for ii,p in enumerate(self.particles):
            X = np.vstack((r[:,ix:ix+p.nparticles].T,
                           v[:,ix:ix+p.nparticles].T))
            o = Orbit(t, X,
                      names=p.names,
                      units=p._internal_units,
                      meta=p.meta)
            ix += p.nparticles
            orbits.append(o)

        return orbits