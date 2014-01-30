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
from .. import potential as sp
from ..util import _parse_quantity
from .. import usys

__all__ = ["StreamModel", "StreamModelSampler"]

logger = logging.getLogger(__name__)

class StreamModel(object):

    def __init__(self, potential, lnpargs=(),
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
        self._given_potential_params = dict([(k,v._value)
                                             for k,v in potential.parameters.items()])

        self.parameters = OrderedDict()
        self.parameters['potential'] = OrderedDict()
        self.parameters['particles'] = OrderedDict()
        self.parameters['satellite'] = OrderedDict()
        self.nparameters = 0

        self.lnpargs = lnpargs

        self.true_satellite = true_satellite
        self.true_particles = true_particles

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
        nparticles = true_particles.nparticles
        logger.info("Running with {} particles.".format(nparticles))

        # integration stuff
        t1 = config.get("t1", float(d["t1"]))
        t2 = config.get("t2", float(d["t2"]))
        dt = config.get("dt", -1.)

        # get the potential object specified from the potential subpackage
        Potential = getattr(sp, config["potential"]["class_name"])
        potential = Potential()
        logger.info("Using potential '{}'...".format(config["potential"]["class_name"]))

        # Define the empty model to add parameters to
        model = cls(potential, lnpargs=[t1,t2,dt,1.], # 1. is the inverse temperature
                    true_satellite=true_satellite,
                    true_particles=true_particles)

        # Potential parameters
        if config["potential"]["parameters"] is not None:
            for ii,(name,kwargs) in enumerate(config["potential"]["parameters"].items()):
                a,b = kwargs["a"], kwargs["b"]
                p = getattr(potential, name)
                logger.debug("Prior on {}: Uniform({}, {})".format(name, a, b))

                prior = LogUniformPrior(_parse_quantity(a).decompose(usys).value,
                                           _parse_quantity(b).decompose(usys).value)
                p = ModelParameter(name, prior=prior, truth=potential.parameters[name]._truth)
                model.add_parameter('potential', p)

        # Particle parameters
        if config['particles']['parameters'] is not None:
            particles = d['particles']

            logger.debug("Particle properties added as parameters:")
            if config['particles']['parameters'].has_key('_X'):
                priors = [LogNormalPrior(particles._X[ii],particles._error_X[ii])
                            for ii in range(nparticles)]
                X = ModelParameter('_X', value=particles._X, prior=priors,
                                      truth=true_particles._X)
                model.add_parameter('particles', X)
                logger.debug("\t\t_X - particle 6D positions today")

            if config['particles']['parameters'].has_key('tub'):
                priors = [LogUniformPrior(t2, t1) for ii in range(nparticles)]
                tub = ModelParameter('tub', value=np.zeros(nparticles), prior=priors,
                                        truth=true_particles.tub)
                model.add_parameter('particles', tub)
                logger.debug("\t\ttub - unbinding time of particles")

        # Satellite parameters
        if config['satellite']['parameters'] is not None:
            satellite = d['satellite']

            logger.debug("Satellite properties added as parameters:")
            if config['satellite']['parameters'].has_key('_X'):
                priors = [LogNormalPrior(satellite._X[0],satellite._error_X[0])]
                s_X = ModelParameter('_X', value=satellite._X, prior=priors,
                                        truth=true_satellite._X)
                model.add_parameter('satellite', s_X)
                logger.debug("\t\t_X - satellite 6D position today")

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

                    ### HACK TO INITIALIZE WALKERS NEAR true tub!
                    if param_name == "tub":
                        p0[ii,ix1:ix1+param.size] = np.random.normal(self.true_particles.tub,
                                                                     100.)

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

    def ln_posterior(self, p, *args):
        """ Evaluate the log-posterior at the given parameter vector.

            Parameters
            ----------
            p : array_like
                The vector of model parameter values.
        """

        # TODO: placeholder
        param_dict = self._decompose_vector(p)

        ln_prior = 0.
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                lp = param.prior(param_dict[group_name][param_name])
                ln_prior += lp

        # short-circuit if any prior value is -infinity
        if np.any(np.isinf(ln_prior)):
            return -np.inf

        # get potential
        pparams = self._given_potential_params.copy()
        for k,v in param_dict.get('potential',dict()).items():
            pparams[k] = v

        # heliocentric particle positions and unbinding times
        try:
            p_hel = param_dict['particles']['_X']
        except KeyError:
            p_hel = self.true_particles._X

        try:
            tub = param_dict['particles']['tub']
        except KeyError:
            tub = self.true_particles.tub

        # heliocentric satellite position
        try:
            s_hel = param_dict['satellite']['_X']
        except KeyError:
            s_hel = self.true_satellite._X

        potential = self._potential_class(**pparams)
        ln_like = back_integration_likelihood(args[0], args[1], args[2], # t1, t2, dt
                                              potential, p_hel, s_hel, tub)

        try:
            return (ln_like + np.sum(ln_prior))**args[3]
        except:
            return ln_like + np.sum(ln_prior)

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

