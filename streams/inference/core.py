# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import emcee
import numpy as np
import astropy.units as u

from ..coordinates.frame import galactocentric
from ..dynamics import Particle, ObservedParticle
from ..integrate import ParticleIntegrator
from .parameter import *
from .prior import *
from ..util import get_memory_usage

__all__ = ["StreamModel"]

logger = logging.getLogger(__name__)

class StreamModel(object):

    def __init__(self, potential, satellite, particles,
                 parameters=[]):
        """ ...

            Parameters
            ----------
            ...
        """

        self.potential = potential
        self.satellite = satellite
        self.particles = particles
        self.parameters = parameters

        self.acc = np.zeros((particles.nparticles+1,3))

    def __call__(self, p, *args):
        self.vector = np.array(p)
        return self.ln_posterior(*args)

    @property
    def ndim(self):
        return len(self.sample())

    @property
    def vector(self):
        return np.concatenate(map(np.ravel,
                                  [p.get() for p in self.parameters]))

    @vector.setter
    def vector(self, values):
        ind = 0
        for p in self.parameters:
            if len(p):
                p.set(values[ind:ind+len(p)])
                ind += len(p)
            else:
                p.set(values[ind])
                ind += 1

    def sample(self, size=None):
        if size is None:
            return np.hstack([np.ravel(p.sample()) for p in self.parameters])

        for ii in range(size):
            x = np.hstack([np.ravel(p.sample()) for p in self.parameters])
            try:
                d[ii] = x
            except NameError:
                d = np.zeros((size,) + x.shape)
                d[ii] = x

        return d

    def ln_prior(self):
        ppar = np.concatenate([np.atleast_1d(p.ln_prior())\
                               for p in self.parameters])
        if not np.all(np.isfinite(ppar)):
            return -np.inf

        return np.sum(ppar)

    def ln_likelihood(self, *args):
        """ This is a simplified version of the likelihood laid out by Hogg in
            Bread and Butter (https://github.com/davidwhogg/BreadAndButter/).
            The stars are assumed to come from a Gaussian progenitor,
            described by just two scales -- the tidal radius and
            velocity dispersion.
        """

        #t1,t2,dt = args
        t1,t2,dt,tub_true = args

        # The true positions/velocities of the particles are parameters
        Nparticles = self.particles.nparticles

        particles_gc = self.particles.to_frame(galactocentric)
        satellite_gc = self.satellite.to_frame(galactocentric)

        pi = ParticleIntegrator((particles_gc,satellite_gc),
                                self.potential,
                                args=(Nparticles+1, self.acc))
        particle_orbit,satellite_orbit = pi.run(t1=t1, t2=t2, dt=dt)

        # These are the unbinding time indices for each particle
        # t_idx = np.array([np.argmin(np.fabs(satellite_orbit.t.value - tub)) \
        #                     for tub in self.particles.tub])
        t_idx = np.array([np.argmin(np.fabs(satellite_orbit.t.value - tub)) \
                            for tub in tub_true])

        Ntimesteps  = len(particle_orbit.t)

        sat_var = np.zeros((Ntimesteps,6))
        sat_var[:,:3] = self.potential._tidal_radius(self.satellite.m,
                                            satellite_orbit._X[...,:3])*1.26
        sat_var[:,3:] += self.satellite.v_disp
        cov = sat_var**2

        Sigma = np.array([cov[jj] for ii,jj in enumerate(t_idx)])
        p_x = np.array([particle_orbit._X[jj,ii] \
                            for ii,jj in enumerate(t_idx)])
        s_x = np.array([satellite_orbit._X[jj,0] \
                            for ii,jj in enumerate(t_idx)])

        log_p_x_given_phi = -0.5*np.sum(-2.*np.log(Sigma) +
                            (p_x-s_x)**2/Sigma, axis=1) * abs(dt)

        return np.sum(log_p_x_given_phi)

        # if self.obs_data is not None:
        #     log_p_D_given_x = -0.5*np.sum(-2.*np.log(self.obs_error)\
        #                 + (hel-self.obs_data)**2/self.obs_error**2)
        # else:
        #     log_p_D_given_x = 0.

        # if self.obs_data_sat is not None:
        #     log_p_D_given_x_sat = -0.5*np.sum(-2.*np.log(self.obs_error_sat)\
        #             + (hel_sat-self.obs_data_sat)**2/self.obs_error_sat**2)
        # else:
        #     log_p_D_given_x_sat = 0.

    def ln_posterior(self, *args):
        lp = self.ln_prior()
        if not np.isfinite(lp):
            return -np.inf

        ll = self.ln_likelihood(*args)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
