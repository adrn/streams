# coding: utf-8

""" Class for representing observed dynamical object """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
from scipy.stats import norm
from streamteam.units import galactic

__all__ = ['StreamComponent', 'RewinderPotential']

class StreamComponent(object):

    def __init__(self, data, err=None, **kwargs):
        """ TODO """

        self.data = data
        self.err = err
        self.parameters = kwargs
        self._dist = norm(self.data, self.err)

    def ln_data_prob(self, x):
        """ Compute the (log-)probability of phase-space positions (in heliocentric
            observed coordinates), given the data and uncertainties.

        """
        return self._dist.logpdf(x).sum(axis=0)

        # l,b,d,mul,mub,vr = data.T

        # ln_p_l = -np.log(2*np.pi)  # isotropic
        # ln_p_b = np.log(np.cos(b) / 2.)  # isotropic

        # # distance range 1-200 kpc
        # ln_p_d = np.log((1 / np.log(200./1.))) - np.log(d)
        # ln_p_mul = log_normal(mul, 0., 0.306814 / d)  # 300 km/s at d
        # ln_p_mub = log_normal(mub, 0., 0.306814 / d)  # 300 km/s at d
        # ln_p_vr = log_normal(vr, 0., 0.306814)  # 300 km/s

        # return ln_p_l, ln_p_b, ln_p_d, ln_p_mul, ln_p_mub, ln_p_vr


class RewinderPotential(object):

    def __init__(self, Potential, priors=dict(), fixed_pars=dict()):
        """ TODO """

        self.Potential = Potential
        self.fixed_pars = fixed_pars
        self.priors = priors

    def obj(self, **kwargs):
        """ Given kwargs for the potential parameters being varied, return
            a potential object with the given parameters plus any other
            fixed parameters.
        """
        pars = dict(kwargs.items() + self.fixed_pars.items())
        return self.Potential(*pars, units=galactic)

    def ln_prior(self, **kwargs):
        """ Evaluate the value of the log-prior over the potential parameters. """

        lp = 0.
        for k,prior in self.priors.items():
            lp += prior.logpdf(kwargs[k])

        return lp
