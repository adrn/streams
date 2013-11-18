# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

# Project
from ... import usys
from .rrlyrae import rrl_M_V, rrl_V_minus_I

# Create logger
logger = logging.getLogger(__name__)

__all__ = ["ErrorModel", "SpitzerGaiaErrorModel"]

class ErrorModel(object):

    def _get_error(self, x, err):
        """ """

        if isinstance(err, (int, float)):
            return np.ones(len(x)) * err * x
        elif hasattr(err, "unit"):
            return np.ones(len(x)) * err.decompose(self.units).value
        else:
            raise ValueError("Invalid error specification.")

    def __init__(self, l_err, b_err, D_err, mul_err, mub_err, vr_err, \
                 units=usys, factor=1.):
        """ Defines an error model for simulated observations. The
            attributes can be set with either plain numeric values, or
            astropy Quantity objects. If set with a numeric value (float),
            this is interpreted as a fractional error. E.g. for distance,
            D_err = 0.02 means a 2% distance error.

            Parameters
            ----------
            units : list (optional)
                A list of astropy.units.Unit objects that define the
                system of units.
            factor : float (optional)
                A factor to multiply all errors by.

            Attributes
            ----------
            l_err
            b_err
            D_err
            mul_err
            mub_err
            vr_err
        """

        self.units = units
        self.factor = factor

        self.l_err = l_err
        self.b_err = b_err
        self.D_err = D_err
        self.mul_err = mul_err
        self.mub_err = mub_err
        self.vr_err = vr_err

    def __call__(self, O):
        """ Return an array with same shape as O but with the uncertainty
            in each value in O.

            Parameters
            ----------
            O : array_like
                Should have shape (Nparticles, Ndim) and be in
                heliocentric, Galactic coordinates.
        """

        l,b,D,mul,mub,vr = O.T

        # positional
        l_err = self._get_error(l, self.l_err)
        b_err = self._get_error(b, self.b_err)
        D_err = self._get_error(D, self.D_err)

        # velocity
        mul_err = self._get_error(mul, self.mul_err)
        mub_err = self._get_error(mub, self.mub_err)
        vr_err = self._get_error(vr, self.vr_err)

        O_err = np.array([l_err, b_err, D_err, mul_err, mub_err, vr_err]).T
        return O_err * self.factor

class SpitzerGaiaErrorModel(object):

    def __init__(self, *args, **kwargs):
        """ By default, has observational errors for RR Lyrae as observed
            with distances from Spitzer, proper motions from Gaia, and
            ground-based radial velocities.
        """

        # position
        kwargs["l_err"] = kwargs.get("l_err", 100.*u.microarcsecond)
        kwargs["b_err"] = kwargs.get("b_err", 100.*u.microarcsecond)
        kwargs["D_err"] = kwargs.get("D_err", 0.02)

        # velocity
        kwargs["mul_err"] = kwargs.get("mul_err", None)
        kwargs["mub_err"] = kwargs.get("mub_err", None)
        kwargs["vr_err"] = kwargs.get("vr_err", 10*u.km/u.s)

        super(SpitzerGaiaErrorModel, self).__init__(*args, **kwargs)

    def __call__(self, O):
        l,b,D,mul,mub,vr = O.T

        if self.mul_err is None or self.mub_err is None:
            # assuming [Fe/H] = -0.5 for Sgr
            M_V, dM_V = rrl_M_V(-0.5)
            r_unit = filter(lambda x: x.is_equivalent(u.km), self.units)[0]
            V = apparent_magnitude(M_V, D*r_unit)
            dmu = proper_motion_error(V, rrl_V_minus_I)

        ml = False
        if self.mul_err is None:
            self.mul_err = dmu
            ml = True

        mb = False
        if self.mub_err is None:
            self.mub_err = dmu
            mb = True

        vv = super(SpitzerGaiaErrorModel, self).__call__(O)

        if ml:
            self.mul_err = None

        if mb:
            self.mub_err = None

        return vv
