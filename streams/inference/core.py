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

__all__ = ["LogUniformPrior", "Parameter", "StreamModel"]

logger = logging.getLogger(__name__)

class LogPrior(object):

    def __call__(self, value):
        return 0.

class LogUniformPrior(LogPrior):

    def __call__(self, value):
        if np.any((value < self.a) | (value > self.b)):
            return -np.inf
        return 0.0

    def __init__(self, a, b):
        """ Return 0 if value is outside of the range
            defined by a < value < b.
        """
        self.a = a
        self.b = b

    def sample(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)

class Parameter(object):

    def __init__(self, target, attr, ln_prior=None):
        self.target = target
        self.attr = attr

        if ln_prior is None:
            ln_prior = LogPrior()
        self._ln_prior = ln_prior

    def __str__(self):
        return "{0}".format(self.attr)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return "<{0} {1}>".format(self.__class__.__name__,
                                  str(self))

    def get(self):
        return getattr(self.target, self.attr)

    def set(self, value):
        setattr(self.target, self.attr, value)

    def ln_prior(self):
        return self._ln_prior(self.get())

    def __len__(self):
        try:
            return len(self.get())
        except TypeError:
            return 0

# TODO: would be nice if I could pass in one object that knows:
#   - star 6d positions in heliocentric
#   - errors on heliocentric data
#   - frame (heliocentric)
class StreamModel(object):

    def __init__(self, potential, satellite, true_particles,
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
        self.true_particles = true_particles
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

    def ln_prior(self):
        ppar = [p.ln_prior() for p in self.parameters]
        if not np.all(np.isfinite(ppar)):
            return -np.inf

        return 0.

    def ln_likelihood(self, *args):
        """ This is a simplified version of the likelihood laid out by D. Hogg in
            Bread and Butter (https://github.com/davidwhogg/BreadAndButter/). The
            stars are assumed to come from a Gaussian progenitor, described by
            just two scales -- the tidal radius and velocity dispersion.
        """

        t1, t2, dt = args

        # The true positions/velocities of the particles are parameters
        Nparticles = len(self.true_particles)
        x = self.true_particles._X
        hel = _gc_to_hel(x)

        # These are the unbinding times for each particle
        t_idx = -self.true_particles.tub

        acc = np.zeros((Nparticles+1,3))
        s,p = satellite_particles_integrate(self.satellite,
                                        self.true_particles,
                                        self.potential,
                                        potential_args=(Nparticles+1, acc),
                                        time_spec=dict(t1=t1, t2=t2, dt=dt))

        Ntimesteps  = p._X.shape[0]

        sat_var = np.zeros((Ntimesteps,6))
        sat_var[:,:3] = self.potential._tidal_radius(self.satellite._m, s._r)*1.26
        sat_var[:,3:] += sat._v_disp
        cov = sat_var**2

        Sigma = np.array([cov[jj] for ii,jj in enumerate(t_idx)])
        p_x = np.array([p._X[jj,ii] for ii,jj in enumerate(t_idx)])
        s_x = np.array([s._X[jj,0] for ii,jj in enumerate(t_idx)])
        log_p_x_given_phi = -0.5*np.sum(-2.*np.log(Sigma) +
                            (p_x-s_x)**2/Sigma, axis=1) * abs(dt)

        log_p_D_given_x = -0.5*np.sum(-2.*np.log(self.obs_error) + \
                                      (hel-self.obs_data)**2/self.obs_error**2, axis=1)

        return np.sum(log_p_D_given_x + log_p_x_given_phi)

    def ln_posterior(self, *args):
        lp = self.ln_prior()
        if not np.isfinite(lp):
            return -np.inf

        ll = self.ln_likelihood(*args)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
