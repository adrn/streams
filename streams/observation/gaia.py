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
from .rrlyrae import rrl_M_V, rrl_V_minus_I
from .core import apparent_magnitude
from ..coordinates import gc_to_hel, hel_to_gc

__all__ = ["parallax_error", "proper_motion_error",  \
           "apparent_magnitude", "rr_lyrae_add_observational_uncertainties", \
           "add_uncertainties_to_particles", "V_to_G", \
           "rr_lyrae_observational_errors"]

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

# TODO: generalize this, each dimension should have a function?
class RRLyraeErrorModel(object):

    def __init__(self, units, factor=1.):
        """ TODO: need way to change radial velocity error / distance error """
        self.units = units
        self.factor = factor

    def __call__(self, X):
        """ Compute our canonical errors at a grid of given 6D
            positions, X.

            Parameters
            ----------
            X : array_like
                Should have shape (Nparticles, Ndim) and be in
                heliocentric, Galactic coordinates.
        """
        l,b,D,mul,mub,vr = X.T

        # assuming [Fe/H] = -0.5 for Sgr
        M_V, dM_V = rrl_M_V(-0.5)
        r_unit = filter(lambda x: x.is_equivalent(u.km), self.units)[0]
        V = apparent_magnitude(M_V, D*r_unit)

        # proper motion error, from Gaia:
        dmu = proper_motion_error(V, rrl_V_minus_I).decompose(self.units)
        mul_err = dmu.value
        mub_err = dmu.value

        # flat distance error:
        D_err = 0.02*D

        # fixed radial velocity error:
        vr_err = np.ones(len(l))*(10.*u.km/u.s).decompose(self.units).value

        # negligible..
        l_err = np.ones(len(l))*(10.*u.microarcsecond).to(u.radian).value
        b_err = np.ones(len(l))*(10.*u.microarcsecond).to(u.radian).value

        return np.array([l_err, b_err, D_err, mul_err, mub_err, vr_err]).T * self.factor



# TODO: shit below sucks
def rr_lyrae_add_observational_uncertainties(x,y,z,vx,vy,vz,**kwargs):
    """ Given 3D galactocentric position and velocity, transform to heliocentric
        coordinates, apply observational uncertainty estimates, then transform
        back to galactocentric frame.

        TODO: V-I color and metallicity should be *draws* from distributions.

        Parameters
        ----------
        x,y,z : astropy.units.Quantity
            Positions.
        vx,vy,vz : astropy.units.Quantity
            Velocities.
    """

    if not isinstance(x,u.Quantity) or \
       not isinstance(y,u.Quantity) or \
       not isinstance(z,u.Quantity):
        raise TypeError("Positions must be Astropy Quantity objects!")

    if not isinstance(vx,u.Quantity) or \
       not isinstance(vy,u.Quantity) or \
       not isinstance(vz,u.Quantity):
        raise TypeError("Velocities must be Astropy Quantity objects!")

    # Transform to heliocentric coordinates
    x = x.to(u.kpc)
    y = y.to(u.kpc)
    z = z.to(u.kpc)

    vx = vx.to(u.km/u.s)
    vy = vy.to(u.km/u.s)
    vz = vz.to(u.km/u.s)

    l, b, D, mul, mub, vr = gc_to_hel(x,y,z,vx,vy,vz)
    l_err, b_err, D_err, mul_err, mub_err, vr_err = rr_lyrae_observational_errors(l, b, D, mul, mub, vr)

    # DISTANCE ERROR -- assuming 2% distances from RR Lyrae mid-IR
    if kwargs.has_key("distance_error_percent") and \
        kwargs["distance_error_percent"] is not None:
        D_err = kwargs["distance_error_percent"] / 100. * D
    D += np.random.normal(0., D_err.value)*D.unit

    # RADIAL VELOCITY ERROR -- 10 km/s
    if kwargs.has_key("radial_velocity_error") and \
        kwargs["radial_velocity_error"] is not None:
        vr_err = kwargs["radial_velocity_error"]*np.ones(len(x))

    vr_err = vr_err.to(u.km/u.s)
    vr += np.random.normal(0., vr_err.value, size=len(vr))*vr_err.unit

    # PROPER MOTION ERROR
    if kwargs.has_key("proper_motion_error") and \
        kwargs["proper_motion_error"] is not None:
        dmu = kwargs["proper_motion_error"]
        mul_err = mub_err = dmu*np.ones(len(x))

    if kwargs.has_key("proper_motion_error_frac") and \
        kwargs["proper_motion_error_frac"] is not None:
        prc = kwargs["proper_motion_error_frac"]
        dmu = proper_motion_error(V, rrl_V_minus_I)*prc
        mul_err = mub_err = dmu*np.ones(len(x))

    mul_err = mul_err.to(u.rad/u.s)
    mub_err = mub_err.to(u.rad/u.s)

    mul = mul + np.random.normal(0., mul_err.value, size=len(x))*mul_err.unit
    mub = mub + np.random.normal(0., mub_err.value, size=len(x))*mub_err.unit

    x2,y2,z2,vx2,vy2,vz2 = hel_to_gc(l,b,D,mul,mub,vr)
    return (x2.to(u.kpc), y2.to(u.kpc), z2.to(u.kpc), \
            vx2.to(u.kpc/u.Myr), vy2.to(u.kpc/u.Myr), vz2.to(u.kpc/u.Myr))

def add_uncertainties_to_particles(particles, **kwargs):
    """ Given a Particle object, add RR Lyrae-like uncertainties
        and return a new Particle with the errors.
    """

    x,y,z,vx,vy,vz = rr_lyrae_add_observational_uncertainties(particles.r[:,0],
                                                              particles.r[:,1],
                                                              particles.r[:,2],
                                                              particles.v[:,0],
                                                              particles.v[:,1],
                                                              particles.v[:,2],
                                                              **kwargs)

    new_r = np.zeros_like(particles.r.value)
    new_r[:,0] = x.value
    new_r[:,1] = y.value
    new_r[:,2] = z.value

    new_v = np.zeros_like(particles.v.value)
    new_v[:,0] = vx.value
    new_v[:,1] = vy.value
    new_v[:,2] = vz.value

    return Particle(r=new_r*particles.r.unit,
                    v=new_v*particles.v.unit,
                    m=particles.m,
                    units=particles.units)
