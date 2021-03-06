# coding: utf-8

""" Error models for experiments """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy

# Third-party
import numpy as np
import astropy.units as u

# Project
from .. import usys
from gary.observation import apparent_magnitude
from gary.observation.rrlyrae import M_V, gaia_rv_error, gaia_pm_error

__all__ = ["gaia_spitzer_errors", "current_data_star", "current_data_sat"]

def gaia_spitzer_errors(particles):
    """ Given particles in heliocentric frame, return an array of
        Gaia + Spitzer errors (same shape as particles._X).
    """

    try:
        # assuming [Fe/H] = -0.5 for Sgr
        V = apparent_magnitude(M_V(-0.5), particles["d"]*u.kpc)
    except AttributeError:
        raise ValueError("Particles in wrong reference frame? Has "
                         "coordinates: {}".format(particles.names))
    dlb = 100*u.microarcsecond
    dmu = gaia_pm_error(particles["d"]*u.kpc)

    # angular position error is negligible
    l = dlb.decompose(usys).value
    b = dlb.decompose(usys).value

    # proper motion
    mul = dmu.decompose(usys).value
    mub = dmu.decompose(usys).value

    # distance
    d = 0.02*particles["d"]

    # radial velocity
    vr = (5.*u.km/u.s).decompose(usys).value * np.ones_like(particles["vr"])

    return dict(l=l, b=b, d=d, mul=mul,mub=mub,vr=vr)

def spitzer_current_data_sat(particles):

    dlb = 100*u.microarcsecond
    dmu = 0.2*u.mas/u.yr # http://iopscience.iop.org/1538-3881/139/3/839/pdf/aj_139_3_839.pdf

    # angular position error is negligible
    l = dlb.to(particles["l"].unit)
    b = dlb.to(particles["b"].unit)

    # proper motion
    mul = dmu.to(particles["mul"].unit)
    mub = dmu.to(particles["mub"].unit)

    # distance
    d = 0.02*particles["d"]

    # radial velocity
    vr = 5.*u.km/u.s * np.ones_like(particles["vr"].value)

    errs = dict(l=l, b=b, d=d, mul=mul,mub=mub,vr=vr)
    return errs

def current_data_star(particles):
    dlb = 1000*u.microarcsecond

    # http://adsabs.harvard.edu/abs/2013ApJ...766...79K for individual stars
    dmu = 2.*u.mas/u.yr

    # angular position error is negligible
    l = dlb.decompose(usys).value
    b = dlb.decompose(usys).value

    # proper motion
    mul = dmu.decompose(usys).value
    mub = dmu.decompose(usys).value

    # distance
    d = 0.2*particles["d"] # 20% distances, http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=2004AJ....128..245M&db_key=AST&link_type=ABSTRACT&high=51a39ba3d727427

    # radial velocity
    vr = (5.*u.km/u.s).decompose(usys).value * np.ones_like(particles["vr"])
    # M giant, http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=2004AJ....128..245M&db_key=AST&link_type=ABSTRACT&high=51a39ba3d727427

    return dict(l=l, b=b, d=d, mul=mul,mub=mub,vr=vr)

def current_data_sat(particles):

    dlb = 0.1*u.arcsecond
    dmu = 0.2*u.mas/u.yr # http://iopscience.iop.org/1538-3881/139/3/839/pdf/aj_139_3_839.pdf

    # angular position error is negligible
    l = dlb.decompose(usys).value
    b = dlb.decompose(usys).value

    # proper motion
    mul = dmu.decompose(usys).value
    mub = dmu.decompose(usys).value

    # distance
    d = 0.8 # kpc, http://adsabs.harvard.edu/abs/2009AJ....137.4478K

    # radial velocity
    vr = (10.*u.km/u.s).decompose(usys).value

    errs = dict(l=l, b=b, d=d, mul=mul,mub=mub,vr=vr)
    return errs
