# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from numpy import cos, sin

import astropy.coordinates as coord
import astropy.units as u

__all__ = ["vgsr_to_vhel", "vhel_to_vgsr", "ra_dec_dist_to_xyz", \
           "gc_to_hel", "hel_to_gc", "_gc_to_hel", "_hel_to_gc"]


def vgsr_to_vhel(l, b, v_gsr,
                 v_sun_lsr=[10.,5.25,7.17]*u.km/u.s,
                 v_circ=220*u.km/u.s):
    """ Convert a velocity from the Galactic standard of rest (GSR) to
        heliocentric radial velocity.

        Parameters
        ----------

    """

    try:
        v_lsr = v_gsr - v_circ * sin(l.radian) * cos(b.radian)
    except AttributeError:
        raise AttributeError("All inputs must be Quantity objects")

    # velocity correction for Sun relative to LSR
    v_correct = v_sun_lsr[0]*cos(b.radian)*cos(l.radian) + \
                v_sun_lsr[1]*cos(b.radian)*sin(l.radian) + \
                v_sun_lsr[2]*sin(b.radian)
    v_hel = v_lsr - v_correct

    return v_hel

def vhel_to_vgsr(l, b, v_hel,
                 v_sun_lsr=[10.,5.25,7.17]*u.km/u.s,
                 v_circ=220*u.km/u.s):
    """ Convert a velocity from a heliocentric radial velocity to
        the Galactic center of rest.

        Parameters
        ----------

    """
    try:
        v_lsr = v_hel + v_circ * sin(l.radian) * cos(b.radian)
    except AttributeError:
        raise AttributeError("All inputs must be Quantity objects")

    # velocity correction for Sun relative to LSR
    v_correct = v_sun_lsr[0]*cos(b.radian)*cos(l.radian) + \
                v_sun_lsr[1]*cos(b.radian)*sin(l.radian) + \
                v_sun_lsr[2]*sin(b.radian)
    v_gsr = v_lsr + v_correct

    return v_gsr

def ra_dec_dist_to_xyz(ra, dec, dist):
    """ Convert an ra, dec, and distance to a Galactocentric X,Y,Z """

    XYZ = np.zeros((len(ra), 3))
    for ii,(r,d,D) in enumerate(zip(ra, dec, dist)):
        icrs = coord.ICRSCoordinates(r.value, d.value,
                                     unit=(r.unit, d.unit),
                                     distance=coord.Distance(D))
        gal = icrs.galactic
        XYZ[ii,0] = gal.x - 8.
        XYZ[ii,1] = gal.y
        XYZ[ii,2] = gal.z

    return XYZ

def __radial_velocity(r, v):
    """ Compute the radial velocity in the heliocentric frame.

        DON'T USE

    """

    if r.ndim < 2:
        r = r[np.newaxis]

    if v.ndim < 2:
        v = v[np.newaxis]

    # the sun's velocity and position
    v_circ = 220.
    v_sun = np.array([[0., v_circ, 0]]) # km/s
    v_sun += np.array([[9, 11., 6.]]) # km/s
    r_sun = np.array([[-8., 0, 0]])

    # object's distance in relation to the sun(observed radius)
    r_rel = r - r_sun
    R_obs = np.sqrt(np.sum(r_rel**2, axis=-1))[:,np.newaxis]
    r_hat = r_rel / R_obs

    v_rel = v - v_sun
    v_hel = np.sum((v_rel*r_hat), axis=-1)[:,np.newaxis]

    return np.squeeze(v_hel)

def gc_to_hel(x,y,z,vx,vy,vz,
              Rsun=8.*u.kpc, Vcirc=220.*u.km/u.s):
    # transform to heliocentric cartesian
    x = x + Rsun
    vy = vy - Vcirc # don't use -= or +=!!!

    # transform from cartesian to spherical
    d = np.sqrt(x**2 + y**2 + z**2)
    l = np.arctan2(y, x)
    b = np.pi/2.*u.rad - np.arccos(z/d)

    # transform cartesian velocity to spherical
    d_xy = np.sqrt(x**2 + y**2)
    vr = (vx*x + vy*y + vz*z) / d # velocity
    omega_l = -(vx*y - x*vy) / d_xy**2 # angular velocity
    omega_b = -(z*(x*vx + y*vy) - d_xy**2*vz) / (d**2 * d_xy) # angular velocity

    mul = (omega_l.decompose()*u.rad).to(u.milliarcsecond / u.yr)
    mub = (omega_b.decompose()*u.rad).to(u.milliarcsecond / u.yr)

    return l,b,d,mul,mub,vr

def hel_to_gc(l,b,d,mul,mub,vr,
              Rsun=8.*u.kpc,Vcirc=220.*u.km/u.s):
    # transform from spherical to cartesian
    x = d*np.cos(b)*np.cos(l)
    y = d*np.cos(b)*np.sin(l)
    z = d*np.sin(b)

    # transform spherical velocity to cartesian
    omega_l = -mul.to(u.rad/u.s).value/u.s
    omega_b = -mub.to(u.rad/u.s).value/u.s

    vx = x/d*vr + y*omega_l + z*np.cos(l)*omega_b
    vy = y/d*vr - x*omega_l + z*np.sin(l)*omega_b
    vz = z/d*vr - d*np.cos(b)*omega_b

    x -= Rsun
    vy += Vcirc

    return x,y,z,vx,vy,vz

def _gc_to_hel(X):
    """ Assumes Galactic units: kpc, Myr, radian, M_sun """

    Rsun = 8.
    Vcirc = 0.224996676312

    x,y,z,vx,vy,vz = X.T

    # transform to heliocentric cartesian
    x = x + Rsun
    vy = vy - Vcirc # don't use -= or +=!!!

    # transform from cartesian to spherical
    d = np.sqrt(x**2 + y**2 + z**2)
    l = np.arctan2(y, x)
    b = np.pi/2. - np.arccos(z/d)

    # transform cartesian velocity to spherical
    d_xy = np.sqrt(x**2 + y**2)
    vr = (vx*x + vy*y + vz*z) / d # kpc/Myr
    mul = -(vx*y - x*vy) / d_xy**2 # rad / Myr
    mub = -(z*(x*vx + y*vy) - d_xy**2*vz) / (d**2 * d_xy) # rad / Myr

    O = np.array([l,b,d,mul,mub,vr]).T
    return O

def _hel_to_gc(O):
    """ Assumes Galactic units: kpc, Myr, radian, M_sun """

    Rsun = 8.
    Vcirc = 0.224996676312

    l,b,d,mul,mub,vr = O.T

    # transform from spherical to cartesian
    x = d*np.cos(b)*np.cos(l)
    y = d*np.cos(b)*np.sin(l)
    z = d*np.sin(b)

    # transform spherical velocity to cartesian
    mul = -mul
    mub = -mub

    vx = x/d*vr + y*mul + z*np.cos(l)*mub
    vy = y/d*vr - x*mul + z*np.sin(l)*mub
    vz = z/d*vr - d*np.cos(b)*mub

    x -= Rsun
    vy += Vcirc

    X = np.array([x,y,z,vx,vy,vz])
    return X.T