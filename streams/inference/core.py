# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
from collections import OrderedDict

# Third-party
import emcee
import numpy as np
import astropy.units as u

__all__ = ["StreamModel"]

logger = logging.getLogger(__name__)

class StreamModel(object):

    def __init__(self, potential):
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

    def ln_posterior(self, p, *args):
        """ """

        t1, t2, dt = args

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
        p_hel = derp['particles']['_X']
        tub = derp['particles']['tub']

        # heliocentric satellite position
        s_hel = derp['satellite']['_X']

        potential = self._potential_class(**pparams)
        ln_like = back_integration_likelihood(t1, t2, dt, potential,
                                              p_hel, s_hel, tub)

        return ln_like + np.sum(ln_prior)

    def __call__(self, p, *args):
        return self.ln_posterior(p, *args)

    def run(self, Nsteps):
        pass