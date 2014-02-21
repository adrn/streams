# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
from collections import OrderedDict

# Third-party
from emcee import EnsembleSampler, PTSampler
import numpy as np
import astropy.units as u

# Project
from .back_integrate import back_integration_likelihood
from ..io import read_hdf5
from .parameter import ModelParameter
from .prior import *
from ..util import _parse_quantity
from .. import usys

__all__ = ["StreamModel", "StreamModelSampler"]

logger = logging.getLogger(__name__)

class LogCoordinatePrior(LogUniformPrior):

    def __init__(self, name):
        """ Return 0 if value is outside of the range
            defined by a < value < b.
        """
        if name == 'l':
            self.a = (0.*u.deg).decompose(usys).value
            self.b = (360.*u.deg).decompose(usys).value
        elif name == 'b':
            self.a = (-90.*u.deg).decompose(usys).value
            self.b = (90.*u.deg).decompose(usys).value
        elif name == 'd':
            self.a = (5.*u.kpc).decompose(usys).value
            self.b = (150.*u.kpc).decompose(usys).value
        elif name == 'mul':
            self.a = (-100.*u.mas/u.yr).decompose(usys).value
            self.b = (100.*u.mas/u.yr).decompose(usys).value
        elif name == 'mub':
            self.a = (-100.*u.mas/u.yr).decompose(usys).value
            self.b = (100.*u.mas/u.yr).decompose(usys).value
        elif name == 'vr':
            self.a = (-1000.*u.km/u.s).decompose(usys).value
            self.b = (1000.*u.km/u.s).decompose(usys).value

class StreamModel(object):

    def __init__(self, potential, lnpargs=(),
                 satellite=None, particles=None,
                 true_satellite=None, true_particles=None):
        """ Model for tidal streams that uses backwards integration to Rewind
            the positions of stars.

            Parameters
            ----------
            Potential : streams.potential.Potential
                The potential to fit.
            lnpargs : iterable
                Other arguments to the posterior function.
            true_satellite : streams.dynamics.Particle
            true_particles : streams.dynamics.Particle
        """
        self._potential_class = potential.__class__
        self._given_potential_params = potential._parameter_dict.copy()
        self._potential = self._potential_class(**self._given_potential_params)
        #dict([(k,v._value) for k,v in potential.parameters.items()])

        self.parameters = OrderedDict()
        self.parameters['potential'] = OrderedDict()
        self.parameters['particles'] = OrderedDict()
        self.parameters['satellite'] = OrderedDict()
        self.nparameters = 0

        self.lnpargs = lnpargs

        self.satellite = satellite
        self.particles = particles
        self.true_satellite = true_satellite
        self.true_particles = true_particles

        self._prior_cache = dict()

    @classmethod
    def from_config(cls, config):
        """ Construct a StreamModel from a configuration dictionary.
            Typically comes from a YAML file via `streams.io.read_config`.

            Parameters
            ----------
            config : dict
        """
        # load satellite and particle data
        logger.debug("Reading particle/satellite data from:\n\t{}".format(config["data_file"]))
        d = read_hdf5(config["data_file"],
                      nparticles=config.get('nparticles', None),
                      particle_idx=config.get('particle_idx', None))

        true_particles = d['true_particles']
        true_satellite = d['true_satellite']
        satellite = d.get('satellite', None)
        particles = d.get('particles', None)
        nparticles = true_particles.nparticles
        logger.info("Running with {} particles.".format(nparticles))

        # integration stuff
        t1 = config.get("t1", float(d["t1"]))
        t2 = config.get("t2", float(d["t2"]))
        dt = config.get("dt", -1.)

        # get the potential object specified from the potential subpackage
        from .. import potential as sp
        Potential = getattr(sp, config["potential"]["class_name"])
        potential = Potential()
        logger.info("Using potential '{}'...".format(config["potential"]["class_name"]))

        # Define the empty model to add parameters to
        model = cls(potential, lnpargs=[t1,t2,dt],
                    particles=particles,
                    satellite=satellite,
                    true_satellite=true_satellite,
                    true_particles=true_particles)

        # Potential parameters
        if config["potential"]["parameters"]:
            for ii,name in enumerate(config["potential"]["parameters"]):
                p = potential.parameters[name].copy()
                logger.debug("Prior on {}: Uniform({}, {})".format(name, p._prior.a, p._prior.b))
                model.add_parameter('potential', p)

        # Particle parameters
        if config['particles']['parameters']:
            logger.debug("Particle properties added as parameters:")
            for ii,name in enumerate(config["particles"]["parameters"]):
                # prior = LogNormalPrior(particles[name].decompose(usys).value,
                #                        particles.errors[name].decompose(usys).value)
                priors = [LogCoordinatePrior(name) for ii in range(nparticles)]
                X = ModelParameter(name,
                                   value=particles[name].decompose(usys).value,
                                   prior=priors,
                                   truth=true_particles[name].decompose(usys).value)
                model.add_parameter('particles', X)
                logger.debug("\t\t{}".format(name))

        # Satellite parameters
        if config['satellite']['parameters']:
            logger.debug("Satellite properties added as parameters:")
            for ii,name in enumerate(config["satellite"]["parameters"]):
                # prior = LogNormalPrior(satellite[name].decompose(usys).value,
                #                        satellite.errors[name].decompose(usys).value)
                X = ModelParameter(name,
                                   value=satellite[name].decompose(usys).value,
                                   prior=LogCoordinatePrior(name),
                                   truth=true_satellite[name].decompose(usys).value)
                model.add_parameter('satellite', X)
                logger.debug("\t\t{}".format(name))

            # if config['satellite']['parameters'].has_key('logm0'):
            #     prior = LogUniformPrior(14, 23) # 2.5e6 to 2.5e10
            #     logm0 = ModelParameter('logm0', value=19., prior=prior,
            #                             truth=np.log(true_satellite.mass))
            #     model.add_parameter('satellite', logm0)
            #     logger.debug("\t\tlogm0 - log of satellite mass today")

        return model

    def add_parameter(self, parameter_group, parameter):
        """ Add a parameter to the model in the specified parameter group.

            Parameters
            ----------
            parameter_group : str
                Name of the parameter group to add this parameter to.
            parameter : streams.inference.ModelParameter
                The actual parameter.
        """

        # TODO: need a ModelParameter object -- needs to have a .size, .shape
        if not self.parameters.has_key(parameter_group):
            # currently only supports pre-specified parameter groups
            raise KeyError("Invalid parameter group '{}'".format(parameter_group))

        self.parameters[parameter_group][parameter.name] = parameter.copy()
        self.nparameters += parameter.size

    @property
    def truths(self):
        """ Returns an array of the true values of all parameters in the model """

        true_p = np.array([])
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                t = param.truth
                if t is None:
                    t = [None]*param.size
                true_p = np.append(true_p, np.ravel(t))

        return true_p

    def sample_priors(self, size=None):
        """ Draw samples from the priors over the model parameters.

            Parameters
            ----------
            size : int
                Number of samples to draw.
        """

        sz = size if size is not None else 1
        p0 = np.zeros((sz, self.nparameters))
        for ii in range(sz):
            ix1 = 0
            for group_name,group in self.parameters.items():
                for param_name,param in group.items():
                    if param.size > 1:
                        p0[ii,ix1:ix1+param.size] = np.ravel(param.sample().T)
                    else:
                        p0[ii,ix1:ix1+param.size] = np.ravel(param.sample())

                    if param_name in ['l','b','d','mul','mub','vr']:
                        if group_name == 'particles':
                            p = self.particles
                        elif group_name == 'satellite':
                            p = self.satellite

                        try:
                            prior = self._prior_cache[(group_name,param_name)]
                        except KeyError:
                            prior = LogNormalPrior(p[param_name].decompose(usys).value,
                                                   p.errors[param_name].decompose(usys).value)
                            self._prior_cache[(group_name,param_name)] = prior

                        p0[ii,ix1:ix1+param.size] = np.ravel(prior.sample().T)

                    ix1 += param.size

        return np.squeeze(p0)

    def _decompose_vector(self, p):
        """ Turns a vector of parameter values, e.g. from MCMC, and turns it into
            a dictionary of parameters.

            Parameters
            ----------
            p : array_like
                The vector of model parameter values.
        """
        d = OrderedDict()

        ix1 = 0
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                try:
                    d[group_name][param_name] = p[ix1:ix1+param.size]
                except:
                    d[group_name] = OrderedDict()
                    d[group_name][param_name] = p[ix1:ix1+param.size]

                if d[group_name][param_name].size > 1:
                    d[group_name][param_name] = d[group_name][param_name].reshape(param.shape)
                else:
                    d[group_name][param_name] = np.squeeze(d[group_name][param_name])

                ix1 += param.size

        return d

    def label_flatchain(self, flatchain):
        """ Turns a flattened MCMC chain (e.g., sampler.flatchain from emcee) and
            turns it into a dictionary of parameter samples.

            Parameters
            ----------
            flatchain : array_like
                The flattened MCMC chain of parameter values. Should have shape (nsteps, ndim).
        """
        d = OrderedDict()

        nsamples,ndim = flatchain.shape
        # make sure ndim == nparameters
        if ndim != self.nparameters:
            raise ValueError("Flatchain ndim != model nparameters")

        ix1 = 0
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                try:
                    d[group_name][param_name] = flatchain[:,ix1:ix1+param.size]
                except:
                    d[group_name] = OrderedDict()
                    d[group_name][param_name] = flatchain[:,ix1:ix1+param.size]

                if param.size > 1:
                    shp = (nsamples,) + param.shape
                    d[group_name][param_name] = d[group_name][param_name].reshape(shp)

                ix1 += param.size

        return d

    def ln_prior(self, param_dict, *args):
        ln_prior = 0.
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                lp = param.prior(param_dict[group_name][param_name])
                ln_prior += lp

        return ln_prior

    def ln_likelihood(self, param_dict, *args):
        """ Evaluate the log-posterior at the given parameter set.

            Parameters
            ----------
            param_dict : dict
                The dictionary of model parameter values.
        """

        # get potential
        pparams = self._given_potential_params.copy()
        for k,v in param_dict.get('potential',dict()).items():
            pparams[k] = v

        # heliocentric particle positions
        p_hel = []
        # TODO: only use true_particles if none of these are specified.
        #       otherwise, fix at the "observed" position, e.g.
        # if len(param_dict['particles']) == 0:
        #     ...true positions...
        # else:
        #     for k in ['l','b','d','mul','mub','vr']:
        #         try:
        #             p_hel.append(param_dict['particles'][k])
        #         except KeyError:
        #             p_hel.append(self.particles[k].decompose(usys).value)

        data_like = 0.
        for k in ['l','b','d','mul','mub','vr']:
            try:
                p_hel.append(param_dict['particles'][k])
                fn = self._prior_cache[('particles',k)]
                data_like += fn(param_dict['particles'][k])
            except KeyError:
                p_hel.append(self.true_particles[k].decompose(usys).value)
        p_hel = np.vstack(p_hel).T

        # heliocentric satellite positions
        s_hel = []
        for k in ['l','b','d','mul','mub','vr']:
            try:
                s_hel.append(param_dict['satellite'][k])
                fn = self._prior_cache[('satellite',k)]
                data_like += fn(param_dict['satellite'][k])
            except KeyError:
                s_hel.append(self.true_satellite[k].decompose(usys).value)
        s_hel = np.vstack(s_hel).T

        # satellite mass
        # try:
        #     logm0 = param_dict['satellite']['logm0']
        # except KeyError:
        #     logm0 = np.log(self.true_satellite.mass)
        logm0 = np.log(self.true_satellite.mass)

        # HACK
        tail_bit = self.true_particles.tail_bit

        # TODO: don't create new potential each time, just modify _parameter_dict?
        potential = self._potential_class(**pparams)
        t1, t2, dt = args[:3]
        ln_like = back_integration_likelihood(t1, t2, dt,
                                              potential, p_hel, s_hel,
                                              logm0,
                                              self.true_satellite.vdisp,
                                              tail_bit)

        return np.sum(ln_like) + data_like

    def ln_posterior(self, p, *args):

        param_dict = self._decompose_vector(p)

        ln_prior = self.ln_prior(param_dict, *args)

        # short-circuit if any prior value is -infinity
        if np.any(np.isinf(ln_prior)):
            return -np.inf

        ln_like = self.ln_likelihood(param_dict, *args)

        if np.any(np.isnan(ln_like)) or np.any(np.isnan(ln_prior)):
            return -np.inf

        try:
            return np.sum(ln_like)*args[3] + np.sum(ln_prior)
        except:
            return np.sum(ln_like) + np.sum(ln_prior)

    def __call__(self, p):
        # TODO: each call, adjust temperature according to self.annealing?
        return self.ln_posterior(p, *self.lnpargs)

class StreamModelSampler(EnsembleSampler):

    def __init__(self, model, nwalkers=None, pool=None):
        """ """

        if nwalkers is None:
            nwalkers = model.nparameters*2 + 2

        super(StreamModelSampler, self).__init__(nwalkers, model.nparameters, model,
                                                 pool=pool)

