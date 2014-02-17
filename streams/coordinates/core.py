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

__all__ = ["vgsr_to_vhel", "vhel_to_vgsr", \
           "gc_to_hel", "hel_to_gc", "_gc_to_hel", "_hel_to_gc"]

vcirc = 220.*u.km/u.s
vlsr = [10., 5.25, 7.17]*u.km/u.s
R_sun = 8.*u.kpc

def vgsr_to_vhel(l, b, vgsr,
                 vcirc=vcirc, vlsr=vlsr):
    """ Convert a velocity from the Galactic standard of rest (GSR) to
        a barycentric radial velocity.

        Parameters
        ----------
        l : astropy.coordinates.Angle, astropy.units.Quantity
            Galactic longitude.
        b : astropy.coordinates.Angle, astropy.units.Quantity
            Galactic latitude.
        vgsr : astropy.units.Quantity-like
            GSR velocity.
        vcirc :
    """
    l = coord.Angle(l)
    b = coord.Angle(b)

    # compute the velocity relative to the LSR
    lsr = vgsr - vcirc*sin(l)*cos(b)

    # velocity correction for Sun relative to LSR
    v_correct = vlsr[0]*cos(b)*cos(l) + \
                vlsr[1]*cos(b)*sin(l) + \
                vlsr[2]*sin(b)
    vhel = lsr - v_correct

    return vhel

def vhel_to_vgsr(l, b, vhel,
                 vcirc=vcirc,
                 vlsr=vlsr):
    """ Convert a velocity from a heliocentric radial velocity to
        the Galactic center of rest.

        Parameters
        ----------

    """
    l = coord.Angle(l)
    b = coord.Angle(b)

    lsr = vhel + vcirc*sin(l)*cos(b)

    # velocity correction for Sun relative to LSR
    v_correct = vlsr[0]*cos(b)*cos(l) + \
                vlsr[1]*cos(b)*sin(l) + \
                vlsr[2]*sin(b)
    vgsr = lsr + v_correct

    return vgsr

def gc_to_hel(x,y,z,vx,vy,vz,
              vcirc=vcirc,
              vlsr=vlsr,
              R_sun=R_sun):

    # transform to heliocentric cartesian
    x = x + R_sun
    vy = vy - vcirc # don't use -= or +=!!!

    # correct for motion of LSR
    vx = vx - vlsr[0]
    vy = vy - vlsr[1]
    vz = vz - vlsr[2]

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
              vcirc=vcirc,
              vlsr=vlsr,
              R_sun=R_sun):
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

    x = x - R_sun
    vy = vy + vcirc

    # correct for motion of LSR
    vx = vx + vlsr[0]
    vy = vy + vlsr[1]
    vz = vz + vlsr[2]

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