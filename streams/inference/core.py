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

__all__ = ["LogUniformPrior", "Parameter", "StreamModel"]

logger = logging.getLogger(__name__)

class LogUniformPrior(object):

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

    def __init__(self, value, name="", ln_prior=None,
                 range=(None, None), latex=None):

        self.value = value
        self.name = name
        self._ln_prior = ln_prior
        self._range = range
        self.fixed = False

        if self._ln_prior is None:
            self._ln_prior = LogUniformPrior(*self._range)

    def ln_prior(self):
        return self._ln_prior(self.value)

    def sample(self, size=1):
        return self._ln_prior.sample(size=size)

    def __repr__(self):
        return "<Parameter {0}={1}>".format(self.name, self.value)

    def _repr_latex_(self):
        return self.latex if self.latex else self.__repr__()

class StreamModel(object):

    def __init__(self, potential, satellite, particles):
        """ ...

            Parameters
            ----------
            ...
        """

        # ARGH -- no, cause when p updates self.vector, has to feed back
        #   in to the satellite object...
        # - the likelihood function will be defined in here...so what do I need?
        #satellite_6d = Parameter(satellite._X, range=(-300., 300.))
        #satellite_shape = Parameter(, range=(-300., 300.))

        self.satellite = satellite


        self.parameters = []
        for p in potential.parameters.values():
            if not p.fixed:
                self.parameters.append(p)

    def __call__(self, p):
        self.vector = p
        return self.ln_posterior()

    def ln_likelihood(self, t1, t2, dt):
        """ This is a simplified version of the likelihood laid out by D. Hogg in
            Bread and Butter (https://github.com/davidwhogg/BreadAndButter/). The
            stars are assumed to come from a Gaussian progenitor, described by
            just two scales -- the tidal radius and velocity dispersion.
        """

        # First need to pull apart the parameters p -- first few are the
        #   potential parameters, then the true position of the stars, then
        #   the time the stars came unbound from their progenitor.
        Nparticles,Ndim = data.shape
        Nparams = len(potential_params)
        dt = -1.

        # Use the specified Potential class and parameters
        potential_params = dict(zip(potential_params, p[:Nparams]))
        potential = Potential(**potential_params)

        # These are the true positions/velocities of the particles, which we
        #   add as parameters in the model
        x = np.array(p[Nparams:Nparams+(Nparticles*6)]).reshape(Nparticles,6)
        hel = _gc_to_hel(x)

        # These are the unbinding times for each particle
        t_idx = [int(pp) for pp in p[Nparams+(Nparticles*6):]]

        # A Particle object for the true positions of the particles -- not great...
        particles = Particle(x[:,:3]*u.kpc, x[:,3:]*u.kpc/u.Myr, 0.*u.M_sun)

        acc = np.zeros((Nparticles+1,3))
        s,p = satellite_particles_integrate(satellite, particles, potential,
                                            potential_args=(Nparticles+1, acc),
                                            time_spec=dict(t1=t1, t2=t2, dt=dt))

        Ntimesteps  = p._X.shape[0]

        sat_var = np.zeros((Ntimesteps,6))
        sat_var[:,:3] = potential._tidal_radius(satellite._m, s._r) * 1.26
        sat_var[:,3:] += 0.0083972030362941957 #v_disp # kpc/Myr for 2.5E7
        cov = sat_var**2

        Sigma = np.array([cov[jj] for ii,jj in enumerate(t_idx)])
        p_x = np.array([p._X[jj,ii] for ii,jj in enumerate(t_idx)])
        s_x = np.array([s._X[jj,0] for ii,jj in enumerate(t_idx)])
        log_p_x_given_phi = -0.5*np.sum(-2.*np.log(Sigma) +
                            (p_x-s_x)**2/Sigma, axis=1) * abs(dt)

        log_p_D_given_x = -0.5*np.sum(-2.*np.log(data_errors) + \
                                      (hel-data)**2/data_errors**2, axis=1)

        return np.sum(log_p_D_given_x + log_p_x_given_phi)

    def ln_prior(self):
        lp = self.planetary_system.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        pp = [l() for l in self.lnpriors]
        if not np.all(np.isfinite(pp)):
            return -np.inf
        ppar = [p.lnprior() for p in self.parameters]
        if not np.all(np.isfinite(ppar)):
            return -np.inf
        return lp + np.sum(pp) + np.sum(ppar)

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

    # ---

    def ln_prior(self, p):
        """ Evaluate the prior functions """

        _sum = 0.
        for ii,param in enumerate(self.parameters):
            if self._prior_funcs.has_key(param):
                _sum += self._prior_funcs[param](p[ii])
            else:
                lo,hi = self.parameter_bounds[param]
                if p[ii] < lo or p[ii] > hi:
                    return -np.inf

        return _sum

    def ln_posterior(self, p):
        return self.ln_prior(p)+self.ln_likelihood(p, *self.likelihood_args)

    def run(self, p0, nsteps, nburn=None, pool=None):
        """ Use emcee to sample from the posterior.

            Parameters
            ----------
            p0 : array
                2D array of starting positions for all walkers.
            nsteps : int (optional)
                Number of steps for each walker to take through
                parameter space.
            burn_in : int (optional)
                Defaults to 1/10 the number of steps.
            pool : multiprocessing.Pool, emcee.MPIPool
                A multiprocessing or MPI pool to pass to emcee for
                wicked awesome parallelization!
        """
        if nburn == None:
            nburn = nsteps // 10

        p0 = np.array(p0)
        nwalkers, ndim = p0.shape

        if ndim != len(self.parameters):
            raise ValueError("Parameter initial conditions must have shape"
                             "(nwalkers,ndim) ({0},{1})".format(nwalkers,
                                len(self.parameters)))

        # make the ensemble sampler
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim,
                                        lnpostfn=self.ln_posterior,
                                        pool=pool)

        logger.debug("About to start walkers...")

        # If a burn-in period requested, run the sampler for 'burn_in'
        #   steps then reset the walkers and use the end positions as
        #   new initial conditions
        if nburn > 0:
            pos, prob, state = sampler.run_mcmc(p0, nburn)
            sampler.reset()
        else:
            pos = p0

        # Run the MCMC sampler and draw nsteps samples per walker
        pos, prob, state = sampler.run_mcmc(pos, nsteps)

        return sampler
