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

__all__ = ["StreamModel", "StreamModelSampler"]

logger = logging.getLogger(__name__)

class StreamModel(object):

    def __init__(self, potential, lnpargs=(),
                 true_satellite=None, true_particles=None):
        """

            Parameters
            ----------
            Potential : streams.Potential
                The potential to fit.
        """
        self._potential_class = potential.__class__
        self._given_potential_params = potential.parameters.copy()

        self.parameters = OrderedDict()
        self.parameters['potential'] = OrderedDict()
        self.parameters['particles'] = OrderedDict()
        self.parameters['satellite'] = OrderedDict()
        self.nparameters = 0

        self.lnpargs = lnpargs

        self.true_satellite = true_satellite
        self.true_particles = true_particles

    def add_parameter(self, parameter_group, parameter):
        """ """

        # TODO: need a ModelParameter object -- needs to have a .size, .shape
        if not self.parameters.has_key(parameter_group):
            # currently only supports pre-specified parameter groups
            raise KeyError("Invalid parameter group '{}'".format(parameter_group))

        self.parameters[parameter_group][parameter.name] = parameter.copy()
        self.nparameters += parameter.size

    @property
    def truths(self):
        """ """
        true_p = np.array([])
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                t = param.truth
                if t is None:
                    t = [None]*param.size
                true_p = np.append(true_p, np.ravel(t))

        return true_p

    def sample_priors(self, size=None):
        """ """

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
                    ix1 += param.size

        return np.squeeze(p0)

    def _decompose_vector(self, p):
        """ """
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
        """ """
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
        """ """

        # TODO: placeholder
        derp = self._decompose_vector(p)

        ln_prior = 0.
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                lp = param.prior(derp[group_name][param_name])
                ln_prior += lp

        # short-circuit if any prior value is -infinity
        if np.any(np.isinf(ln_prior)):
            return -np.inf

        # get potential
        pparams = self._given_potential_params.copy()
        for k,v in derp['potential'].items():
            pparams[k] = v

        # heliocentric particle positions and unbinding times
        try:
            p_hel = derp['particles']['_X']
        except KeyError:
            p_hel = self.true_particles._X

        try:
            tub = derp['particles']['tub']
        except KeyError:
            tub = self.true_particles.tub

        # heliocentric satellite position
        try:
            s_hel = derp['satellite']['_X']
        except KeyError:
            s_hel = self.true_satellite._X

        potential = self._potential_class(**pparams)
        ln_like = back_integration_likelihood(args[0], args[1], args[2], # t1, t2, dt
                                              potential, p_hel, s_hel, tub)

        try:
            return ln_like*args[3] + np.sum(ln_prior)
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

