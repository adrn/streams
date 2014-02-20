# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable

from ..dynamics import Particle
from ..coordinates import gc_to_hel, hel_to_gc
from .rrlyrae import *
from .core import *

__all__ = ["parallax_error", "proper_motion_error", "V_to_G", \
           "gaia_spitzer_errors"]

def V_to_G(V, V_minus_I):
    """ Convert Johnson V to Gaia G-band.

        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """
    return V - 0.0257 - 0.0924*V_minus_I - 0.1623*V_minus_I**2 + 0.009*V_minus_I**3

def parallax_error(V, V_minus_I):
    """ Compute the estimated GAIA parallax error as a function of apparent
        V-band magnitude and V-I color. All equations are taken from the GAIA
        science performance book:

            http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1

        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """

    # GAIA G mag
    g = V - 0.0257 - 0.0924*V_minus_I- 0.1623*V_minus_I**2 + 0.0090*V_minus_I**3
    z = 10**(0.4*(g-15.))

    p = g < 12.
    if isiterable(V):
        if sum(p) > 0:
            z[p] = 10**(0.4*(12. - 15.))
    else:
        if p:
            z = 10**(0.4*(12. - 15.))

    # "end of mission parallax standard"
    # σπ [μas] = (9.3 + 658.1·z + 4.568·z^2)^(1/2) · [0.986 + (1 - 0.986) · (V-IC)]
    dp = np.sqrt(9.3 + 658.1*z + 4.568*z**2) * (0.986 + (1 - 0.986)*V_minus_I) * 1E-6 * u.arcsecond

    return dp

def proper_motion_error(V, V_minus_I):
    """ Compute the estimated GAIA proper motion error as a function of apparent
        V-band magnitude and V-I color. All equations are taken from the GAIA
        science performance book:

            http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1

        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """

    dp = parallax_error(V, V_minus_I)

    # assume 5 year baseline, µas/year
    dmu = dp/u.year

    # too optimistic: following suggests factor 2 more realistic
    #http://www.astro.utu.fi/~cflynn/galdyn/lecture10.html
    # - and Sanjib suggests factor 0.526
    dmu = 0.526*dmu

    return dmu.to(u.radian/u.yr)

def gaia_spitzer_errors(particles):
    """ Given particles in heliocentric frame, return an array of
        Gaia + Spitzer errors (same shape as particles._X).
    """

    # assuming [Fe/H] = -0.5 for Sgr
    M_V, dM_V = rrl_M_V(-0.5)

    try:
        V = apparent_magnitude(M_V, particles["d"])
    except AttributeError:
        raise ValueError("Particles in wrong reference frame? Has "
                         "coordinates: {}".format(particles.names))
    dlb = parallax_error(V, rrl_V_minus_I)
    dmu = proper_motion_error(V, rrl_V_minus_I)

    # angular position error is negligible
    # HACK: extra factor of 100 here
    l = dlb.to(particles["l"].unit)*100
    b = dlb.to(particles["b"].unit)*100

    # proper motion
    mul = dmu.to(particles["mul"].unit)
    mub = dmu.to(particles["mub"].unit)

    # distance
    d = 0.02*particles["d"]

    # radial velocity
    vr = 5.*u.km/u.s * np.ones_like(particles["vr"].value)

    return dict(l=l, b=b, d=d, mul=mul,mub=mub,vr=vr)
