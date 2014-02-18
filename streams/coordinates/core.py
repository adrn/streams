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

__all__ = ["vgsr_to_vhel", "vhel_to_vgsr",
           "gal_xyz_to_hel_lbd", "hel_lbd_to_gal_xyz",
           "gc_to_hel", "hel_to_gc"]

vcirc = 220.*u.km/u.s
vlsr = [10., 5.25, 7.17]*u.km/u.s
xsun = -8.*u.kpc

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
        vgsr : astropy.units.Quantity
            GSR line-of-sight velocity.
        vcirc : astropy.units.Quantity
            Circular velocity of the Sun.
        vlsr : astropy.units.Quantity
            Velocity of the Sun relative to the local standard
            of rest (LSR).

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
                 vcirc=vcirc, vlsr=vlsr):
    """ Convert a velocity from a heliocentric radial velocity to
        the Galactic center of rest.

        Parameters
        ----------
        l : astropy.coordinates.Angle, astropy.units.Quantity
            Galactic longitude.
        b : astropy.coordinates.Angle, astropy.units.Quantity
            Galactic latitude.
        vhel : astropy.units.Quantity
            Barycentric line-of-sight velocity.
        vcirc : astropy.units.Quantity
            Circular velocity of the Sun.
        vlsr : astropy.units.Quantity
            Velocity of the Sun relative to the local standard
            of rest (LSR).

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

def gal_xyz_to_hel_lbd(r, v=None,
                       vcirc=vcirc, vlsr=vlsr,
                       xsun=xsun):
    """ Convert Galactocentric cartesian coordinates to Heliocentric
        spherical coordinates. Uses a right-handed cartesian system,
        typically with the Sun at X ~ -8 kpc.

        Parameters
        ----------
        r : astropy.units.Quantity
            Cartesian x,y,z coordinates. Should have shape (3,N).
        v : astropy.units.Quantity (optional)
            Cartesian velocity components. Sometimes called U,V,W.
            Should have shape (3,N).
        vcirc : astropy.units.Quantity
            Circular velocity of the Sun.
        vlsr : astropy.units.Quantity
            Velocity of the Sun relative to the local standard
            of rest (LSR).
        xsun : astropy.units.Quantity
            Position of the Sun on the Galactic x-axis.
    """
    # unpack positions
    try:
        x,y,z = r
    except ValueError:
        if len(r.shape) > 1 and r.shape[0] > r.shape[1]:
            raise ValueError("Could not unpack positions -- the shape looks"
                             " transposed. Should have shape (3,N).")
        else:
            raise ValueError("Failed to unpack positions with shape {}."
                             " Should have shape (3,N).".format(r.shape))

    # transform to heliocentric cartesian
    x = x - xsun

    # transform from cartesian to spherical
    d = np.sqrt(x**2 + y**2 + z**2)
    l = coord.Angle(np.arctan2(y, x)).wrap_at(360*u.deg).to(u.degree)
    b = coord.Angle(90*u.degree - np.arccos(z/d)).to(u.degree)

    if v is not None:
        if v.shape != r.shape:
            raise ValueError("Shape of velocity should match position.")

        # unpack velocities
        vx,vy,vz = v

        # transform to heliocentric cartesian
        vy = vy - vcirc

        # correct for motion of Sun relative to LSR
        vx = vx - vlsr[0]
        vy = vy - vlsr[1]
        vz = vz - vlsr[2]

        # transform cartesian velocity to spherical
        d_xy = np.sqrt(x**2 + y**2)
        vr = (vx*x + vy*y + vz*z) / d # velocity
        omega_l = -(vx*y - x*vy) / d_xy**2 # angular velocity
        omega_b = -(z*(x*vx + y*vy) - d_xy**2*vz) / (d**2 * d_xy) # angular velocity

        mul = (omega_l.decompose()*u.rad).to(u.milliarcsecond / u.yr)
        mub = (omega_b.decompose()*u.rad).to(u.milliarcsecond / u.yr)

        return (l,b,d), (mul,mub,vr)

    return (l,b,d)


def hel_lbd_to_gal_xyz(lbd, pmvr=None,
                       vcirc=vcirc, vlsr=vlsr,
                       xsun=xsun):
    """ Convert Heliocentric spherical coordinates to Galactocentric
        cartesian coordinates. Uses a right-handed cartesian system,
        typically with the Sun at X ~ -8 kpc.

        Parameters
        ----------
        lbd : tuple (of astropy.units.Quantity objects)
            Galactic longitude, latitude, and distance.
        pmvr : tuple (of astropy.units.Quantity) (optional)
            Proper motion in l, b, and radial velocity.
        vcirc : astropy.units.Quantity
            Circular velocity of the Sun.
        vlsr : astropy.units.Quantity
            Velocity of the Sun relative to the local standard
            of rest (LSR).
        xsun : astropy.units.Quantity
            Position of the Sun on the Galactic x-axis.
    """
    # unpack positions
    try:
        l,b,d = lbd
    except ValueError:
        raise ValueError("Failed to unpack positions.")

    # spherical to cartesian
    x = d*np.cos(b)*np.cos(l)
    y = d*np.cos(b)*np.sin(l)
    z = d*np.sin(b)

    if pmvr is not None:

        # unpack velocities
        mul,mub,vr = pmvr

        omega_l = -mul.to(u.rad/u.s).value/u.s
        omega_b = -mub.to(u.rad/u.s).value/u.s

        vx = x/d*vr + y*omega_l + z*np.cos(l)*omega_b
        vy = y/d*vr - x*omega_l + z*np.sin(l)*omega_b
        vz = z/d*vr - d*np.cos(b)*omega_b

        # transform to galactocentric cartesian
        vy = vy + vcirc

        # correct for motion of Sun relative to LSR
        vx = vx + vlsr[0]
        vy = vy + vlsr[1]
        vz = vz + vlsr[2]

        # transform to galactocentric cartesian
        x = x + xsun

        return np.squeeze(np.vstack((x.value,y.value,z.value)))*x.unit, \
               np.squeeze(np.vstack((vx.value,vy.value,vz.value)))*vx.unit

    else:
        # transform to galactocentric cartesian
        x = x + xsun
        return np.vstack((x.value,y.value,z.value))*x.unit

def gc_to_hel(x,y,z,vx,vy,vz,
              vcirc=vcirc,
              vlsr=vlsr,
              R_sun=8*u.kpc):

    # DEPRECATED

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
              R_sun=8*u.kpc):
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