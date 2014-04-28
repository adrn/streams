# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
from collections import OrderedDict

# Third-party
import numpy as np
import astropy.units as u

# Project
from .. import usys
from .parameter import ModelParameter
from .prior import *

__all__ = ["Model"]

logger = logging.getLogger(__name__)

class Model(object):

    def __init__(self, ln_likelihood=None, ln_likelihood_args=()):
        """ """

        self.parameters = OrderedDict()
        self.nparameters = 0

        if ln_likelihood is None:
            ln_likelihood = lambda *args,**kwargs: 0.

        self.ln_likelihood = ln_likelihood
        self.ln_likelihood_args = ln_likelihood_args

    def add_parameter(self, parameter, parameter_group=None):
        """ Add a parameter to the model.

            Parameters
            ----------
            parameter : ModelParameter
                The parameter instance.
            parameter_group : str (optional)
                Name of the parameter group to add this parameter to.
        """

        if not isinstance(parameter, ModelParameter):
            raise TypeError("Invalid parameter type '{}'".format(type(parameter)))

        if parameter_group is None:
            parameter_group = "main"

        if parameter_group not in self.parameters.keys():
            self.parameters[parameter_group] = OrderedDict()

        self.parameters[parameter_group][parameter.name] = parameter.copy()
        self.nparameters += parameter.size

    def _walk(self, container):
        for group_name,group in container.items():
            for param_name,param in group.items():
                yield group_name, param_name, param

    @property
    def truths(self):
        """ Returns an array of the true values of all parameters in the model """

        true_p = np.array([])
        for group_name,param_name,param in self._walk(self.parameters):
            t = param.truth
            if t is None:
                #t = np.ones(param.size)*np.nan
                t = [None]*param.size
            true_p = np.append(true_p, np.ravel(t))

        return true_p

    def vector_to_parameters(self, p):
        """ Turns a vector of parameter values, e.g. from MCMC, and turns it into
            a dictionary of parameters.

            Parameters
            ----------
            p : array_like
                The vector of model parameter values.
        """
        d = OrderedDict()

        ix1 = 0
        for group_name,param_name,param in self._walk(self.parameters):
            if group_name not in d.keys():
                d[group_name] = OrderedDict()

            par = ModelParameter(name=param_name, value=p[ix1:ix1+param.size],
                                 truth=param.truth, prior=param.prior)
            d[group_name][param_name] = par# p[ix1:ix1+param.size]
            ix1 += param.size

        return d

    def parameters_to_vector(self, parameters):
        """ Turn a parameter dictionary into a parameter vector

            Parameters
            ----------
            param_dict : OrderedDict
        """

        vec = np.array([])
        for group_name,param_name,param in self._walk(parameters):
            p = np.ravel(parameters[group_name][param_name].decompose(usys).value)
            vec = np.append(vec, p)

        return vec

    def ln_prior(self, parameters):
        ln_prior = 0.
        for group_name,param_name,param in self._walk(parameters):
            lp = param.prior(param.value)
            ln_prior += lp

        return ln_prior

    def ln_posterior(self, parameters):
        ln_prior = self.ln_prior(parameters)

        # short-circuit if any prior value is -infinity
        if np.any(np.isinf(ln_prior)):
            return -np.inf

        ln_like = self.ln_likelihood(parameters, *self.ln_likelihood_args)

        if np.any(np.isnan(ln_like)) or np.any(np.isnan(ln_prior)):
            return -np.inf

        return np.sum(ln_like) + np.sum(ln_prior)

    def __call__(self, p):
        parameters = self.vector_to_parameters(p)
        return self.ln_posterior(parameters)

'''
    def sample_priors(self, size=None, start_truth=False):
        """ Draw samples from the priors over the model parameters.

            Parameters
            ----------
            size : int
                Number of samples to draw.
            start_truth : bool (optional)
                Sample centered on true values.
        """

        sz = size if size is not None else 1
        p0 = np.zeros((sz, self.nparameters))
        for ii in range(sz):
            ix1 = 0
            for group_name,group in self.parameters.items():
                for param_name,param in group.items():

                    if param_name in heliocentric.coord_names:
                        err = getattr(self, group_name).errors[param_name].decompose(usys).value
                        if start_truth:
                            val = getattr(self, "true_"+group_name)[param_name]\
                                        .decompose(usys).value
                        else:
                            val = getattr(self, group_name)[param_name]\
                                        .decompose(usys).value

                        if np.any(np.isinf(err)):
                            if param_name in ["mul", "mub", "vr"]:
                                err[np.isinf(err)] = 0.1*val[np.isinf(err)]
                            else:
                                err[np.isinf(err)] = 1.

                        try:
                            prior = self._prior_cache[(group_name,param_name)]
                        except KeyError:
                            prior = LogNormalPrior(val, err)
                            self._prior_cache[(group_name,param_name)] = prior

                        p0[ii,ix1:ix1+param.size] = np.ravel(prior.sample().T)
                        ix1 += param.size

                        continue

                    if start_truth:
                        prior = param._prior
                        if hasattr(prior, 'a') and hasattr(prior, 'b'):
                            mu = np.ravel(param.truth)
                            sigma = mu/100.
                            a = (prior.a - mu) / sigma
                            b = (prior.b - mu) / sigma
                            if param.size > 1:
                                rvs = []
                                for aa,bb,mm,ss in zip(a,b,mu,sigma):
                                    X = truncnorm(aa, bb,
                                                  loc=mm, scale=ss)
                                    rvs.append(X.rvs())
                            else:
                                X = truncnorm(a, b,
                                              loc=mu, scale=sigma)
                                rvs = X.rvs()

                            p0[ii,ix1:ix1+param.size] = np.ravel(rvs)
                        elif hasattr(prior, 'mu'):
                            v = np.random.normal(param.truth, prior.sigma/2.)
                            p0[ii,ix1:ix1+param.size] = np.ravel(v)
                        else:
                            p0[ii,ix1:ix1+param.size] = np.ravel(param.truth)
                    else:
                        if param.size > 1:
                            p0[ii,ix1:ix1+param.size] = np.ravel(param.sample().T)
                        else:
                            p0[ii,ix1:ix1+param.size] = np.ravel(param.sample())

                    ix1 += param.size
        return np.squeeze(p0)

    def ln_likelihood(self, param_dict, *args):
        """ Evaluate the log-posterior at the given parameter set.

            Parameters
            ----------
            param_dict : dict
                The dictionary of model parameter values.
        """

        ######################################################################
        # potential parameters:
        #
        pparams = self._given_potential_params.copy()
        for k,v in param_dict.get('potential',dict()).items():
            pparams[k] = v

        ######################################################################
        # nuisance parameters:

        # particle tail assignment
        try:
            beta = param_dict['particles']['beta']
        except KeyError:
            beta = self.particles.beta.truth
            if np.any(np.isnan(beta)):
                print("True tail assignment was NaN!")
                sys.exit(1)

        # particle unbinding time
        try:
            tub = param_dict['particles']['tub']
        except KeyError:
            tub = self.particles.tub.truth

        # satellite mass
        try:
            logmass = param_dict['satellite']['logmass']
        except KeyError:
            logmass = self.satellite.logmass.truth

        # satellite mass-loss
        try:
            logmdot = param_dict['satellite']['logmdot']
        except KeyError:
            logmdot = self.satellite.logmdot.truth

        # position of effective tidal radius
        try:
            alpha = param_dict['satellite']['alpha']
        except KeyError:
            alpha = self.satellite.alpha.truth

        ######################################################################
        # Heliocentric coordinates
        #

        # first see if any coordinates are parameters. if not, assume we're
        #   running a test with no observational uncertainties
        if param_dict.has_key('particles'):
            param_names = param_dict['particles'].keys()
            num_free = 0
            for pname in param_names:
                if pname in heliocentric.coord_names:
                    num_free += 1
        else:
            num_free = 0

        # Particles
        if num_free == 0:
            # no free coordinate parameters -- use truths
            p_hel = self.true_particles._X
            data_like = 0.
        else:
            p_hel = np.zeros_like(self.particles._X)
            data_like = 0.
            for ii,k in enumerate(heliocentric.coord_names):
                if k in param_dict['particles'].keys():
                    p_hel[:,ii] = param_dict['particles'][k]
                    ll = log_normal(param_dict['particles'][k],
                                    self.particles._X[:,ii],
                                    self.particles._error_X[:,ii])
                else:
                    p_hel[:,ii] = self.particles._X[:,ii]
                    ll = 0.

                if np.any(np.isinf(ll)):
                    ll = 0.

                data_like += ll

            # if any distance is > 150 kpc
            if np.any(p_hel[...,2] > 150.):
                return -np.inf
        p_gc = _hel_to_gc(p_hel)

        # Satellite
        if param_dict.has_key('satellite'):
            param_names = param_dict['satellite'].keys()
            num_free = 0
            for pname in param_names:
                if pname in heliocentric.coord_names:
                    num_free += 1
        else:
            num_free = 0.

        if num_free == 0:
            # no free coordinate parameters -- use truths
            s_hel = self.true_satellite._X
            sat_like = 0.
        else:
            s_hel = np.zeros_like(self.satellite._X)
            sat_like = 0.
            for ii,k in enumerate(heliocentric.coord_names):
                if k in param_dict['satellite'].keys():
                    s_hel[:,ii] = param_dict['satellite'][k]
                    ll = log_normal(param_dict['satellite'][k],
                                    self.satellite._X[:,ii],
                                    self.satellite._error_X[:,ii])
                else:
                    s_hel[:,ii] = self.satellite._X[:,ii]
                    ll = 0.

                if np.any(np.isinf(ll)):
                    ll = 0.

                sat_like += ll

            # if any distance is > 150 kpc
            if np.any(s_hel[...,2] > 150.):
                return -np.inf
        s_gc = _hel_to_gc(s_hel)

        # TODO: don't create new potential each time, just modify _parameter_dict?
        potential = self._potential_class(**pparams)
        t1, t2, dt = args[:3]
        ln_like = back_integration_likelihood(t1, t2, dt,
                                              potential, p_gc, s_gc,
                                              logmass, logmdot,
                                              beta, alpha, tub)

        return np.sum(ln_like + data_like) + np.squeeze(sat_like)



'''