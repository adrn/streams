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

from ..coordinates import _gc_to_hel
from ..integrate.satellite_particles import satellite_particles_integrate
from .parameter import *
from .prior import *

__all__ = ["StreamModel"]

logger = logging.getLogger(__name__)

# TODO: would be nice if I could pass in one object that knows:
#   - star 6d positions in heliocentric
#   - errors on heliocentric data
#   - frame (heliocentric)
class StreamModel(object):

    def __init__(self, potential, satellite, particles,
                 obs_data, obs_error, parameters=[]):
        """ ...

            Parameters
            ----------
            ...
        """
        if obs_data.shape != obs_error.shape:
            raise ValueError("obs_data shape must match obs_errors shape")

        self.potential = potential
        self.satellite = satellite
        self.particles = particles
        self.parameters = parameters
        self.obs_data = obs_data
        self.obs_error = obs_error

    def __call__(self, p, *args):
        self.vector = p
        return self.ln_posterior(*args)

    @property
    def vector(self):
        return np.concatenate(map(np.atleast_1d,
                                  [p.get() for p in self.parameters]))

    @property
    def ndim(self):
        return len(self.sample())

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

        return d

    def ln_prior(self):
        ppar = np.concatenate([np.atleast_1d(p.ln_prior())\
                               for p in self.parameters])
        if not np.all(np.isfinite(ppar)):
            return -np.inf

        return 0.

    def ln_likelihood(self, *args):
        """ This is a simplified version of the likelihood laid out by Hogg in
            Bread and Butter (https://github.com/davidwhogg/BreadAndButter/).
            The stars are assumed to come from a Gaussian progenitor,
            described by just two scales -- the tidal radius and
            velocity dispersion.
        """

        t1, t2, dt = args

        # The true positions/velocities of the particles are parameters
        Nparticles = len(self.particles)
        x = self.particles._X
        hel = _gc_to_hel(x)

        acc = np.zeros((Nparticles+1,3))
        s,p = satellite_particles_integrate(self.satellite,
                                        self.particles,
                                        self.potential,
                                        potential_args=(Nparticles+1, acc),
                                        time_spec=dict(t1=t1, t2=t2, dt=dt))

        # These are the unbinding times for each particle
        t_idx = [np.argmin(np.fabs(s._t - tub)) for tub in self.particles.tub]

        Ntimesteps  = p._X.shape[0]

        sat_var = np.zeros((Ntimesteps,6))
        sat_var[:,:3] = self.potential._tidal_radius(self.satellite._m, s._r)*1.26
        sat_var[:,3:] += self.satellite._v_disp
        cov = sat_var**2

        Sigma = np.array([cov[jj] for ii,jj in enumerate(t_idx)])
        p_x = np.array([p._X[jj,ii] for ii,jj in enumerate(t_idx)])
        s_x = np.array([s._X[jj,0] for ii,jj in enumerate(t_idx)])
        log_p_x_given_phi = -0.5*np.sum(-2.*np.log(Sigma) +
                            (p_x-s_x)**2/Sigma, axis=1) * abs(dt)

        try:
            obs_data = self.particles.obs_data
            obs_error = self.particles.obs_error
            log_p_D_given_x = -0.5*np.sum(-2.*np.log(obs_data) + \
                                (hel-obs_data)**2/obs_error**2, axis=1)
        except AttributeError:
            log_p_D_given_x = 0.

        try:
            obs_data = self.satellite.obs_data
            obs_error = self.satellite.obs_error
            log_p_D_given_x_sat = -0.5*np.sum(-2.*np.log(obs_data) + \
                                (hel-obs_data)**2/obs_error**2)
        except AttributeError:
            log_p_D_given_x_sat = 0.

        return np.sum(log_p_D_given_x + \
                      log_p_D_given_x_sat + \
                      log_p_x_given_phi)

    def ln_posterior(self, *args):
        lp = self.ln_prior()
        if not np.isfinite(lp):
            return -np.inf

        ll = self.ln_likelihood(*args)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
