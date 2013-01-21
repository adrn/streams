# coding: utf-8

""" Transformations between coordinate systems. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

from .core import *

__all__ = ["cartesian_to_spherical", "spherical_to_cartesian", \
           "cartesian_to_cylindrical", "cylindrical_to_cartesian", \
           "spherical_to_cylindrical", "cylindrical_to_spherical"]

# --------------------------------------------------------
#    Cartesian - Spherical
# --------------------------------------------------------

def _cartesian_to_spherical(x,y,z):
    r = np.sqrt(x*x + y*y + z*z)

    if x != 0:
        phi = np.arctan(y / x)
    else:
        phi = 0.

    theta = np.arccos(z / r)
    return dict(r=r, phi=phi, theta=theta)

def _spherical_to_cartesian(r,phi,theta):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return dict(x=x, y=y, z=z)

def cartesian_to_spherical(x=None, y=None, z=None):

    if x is None and y is None and z is None:
        raise ValueError("At least one coordinate must be specified!")

    if x is not None and y is not None and z is None:
        # This is x-y to r-phi : Cartesian to Polar coordinates
        coords = _cartesian_to_spherical(x=x,y=y,z=0.)
        del coords["theta"]
        return coords

    elif x is not None and z is not None and y is None:
        # This is x-z to r-theta : Cartesian to Polar coordinates, where phi=0.
        coords = _cartesian_to_spherical(x=x,y=0.,z=z)
        del coords["phi"]
        return coords

    elif y is not None and z is not None and x is None:
        # This is y-z to r-theta : Cartesian to Polar coordinates, where phi=pi/2
        coords = _cartesian_to_spherical(x=0.,y=y,z=z)
        del coords["phi"]
        return coords

    elif x is not None and y is not None and z is not None:
        return _cartesian_to_spherical(x,y,z)

    elif x is not None:
        # This is just x -> r
        coords = _cartesian_to_spherical(x,y=0.,z=0.)
        del coords["theta"]
        del coords["phi"]
        return coords

    elif y is not None:
        # This is just y -> r
        coords = _cartesian_to_spherical(x=0.,y=y,z=0.)
        del coords["theta"]
        del coords["phi"]
        return coords

    elif z is not None:
        # This is just z -> r
        coords = _cartesian_to_spherical(x=0.,y=0.,z=z)
        del coords["theta"]
        del coords["phi"]
        return coords

    else:
        raise ValueError("Unsupported operation from cartesian to spherical coordinates.")

def spherical_to_cartesian(r, phi=None, theta=None):

    if r is None and phi is None and theta is None:
        raise ValueError("At least one coordinate must be specified!")

    if r is not None and phi is not None and theta is None:
        # This is r-phi to x-y
        coords = _spherical_to_cartesian(r=r, phi=phi, theta=np.pi/2.)
        del coords["z"]
        return coords

    elif r is not None and theta is not None and phi is None:
        # This is r-theta to x-z
        coords = _spherical_to_cartesian(r=r, phi=0., theta=theta)
        del coords["y"]
        return coords

    elif r is not None and phi is not None and theta is not None:
        return _spherical_to_cartesian(r=r, phi=phi, theta=theta)

    else:
        raise ValueError("Unsupported operation from spherical to cartesian coordinates.")


register_transform(CartesianCoordinates, SphericalCoordinates, cartesian_to_spherical, spherical_to_cartesian)

# --------------------------------------------------------
#    Cartesian - Cylindrical
# --------------------------------------------------------

def _cartesian_to_cylindrical(x,y,z):
    r = np.sqrt(x*x + y*y)

    if x != 0:
        phi = np.arctan(y / x)
    else:
        phi = 0.

    return dict(r=r, phi=phi, z=z)

def _cylindrical_to_cartesian(r,phi,z):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z = z
    return dict(x=x, y=y, z=z)

def cartesian_to_cylindrical(x=None, y=None, z=None):

    if x is None and y is None and z is None:
        raise ValueError("At least one coordinate must be specified!")

    if x is not None and y is not None and z is None:
        # This is x-y to r-phi : Cartesian to Polar coordinates
        coords = _cartesian_to_cylindrical(x=x,y=y,z=0.)
        del coords["z"]
        return coords

    elif x is not None and z is not None and y is None:
        # This is x-z to r-z
        coords = _cartesian_to_cylindrical(x=x,y=0.,z=z)
        del coords["phi"]
        return coords

    elif y is not None and z is not None and x is None:
        # This is y-z to r-z
        coords = _cartesian_to_cylindrical(x=0.,y=y,z=z)
        del coords["phi"]
        return coords

    elif x is not None and y is not None and z is not None:
        return _cartesian_to_cylindrical(x,y,z)

    else:
        raise ValueError("Unsupported operation from cartesian to cylindrical coordinates.")

def cylindrical_to_cartesian(r, phi=None, z=None):

    if r is None and phi is None and z is None:
        raise ValueError("At least one coordinate must be specified!")

    if r is not None and phi is not None and z is None:
        # This is r-phi to x-y
        coords = _cylindrical_to_cartesian(r=r, phi=phi, z=0.)
        del coords["z"]
        return coords

    elif r is not None and z is not None and phi is None:
        # This is r-z to x-z
        coords = _cylindrical_to_cartesian(r=r, phi=0., z=z)
        del coords["phi"]
        return coords

    elif r is not None and phi is not None and z is not None:
        return _cylindrical_to_cartesian(r=r, phi=phi, z=z)

    else:
        raise ValueError("Unsupported operation from cylindrical to cartesian coordinates.")


register_transform(CartesianCoordinates, CylindricalCoordinates, cartesian_to_cylindrical, cylindrical_to_cartesian)

# --------------------------------------------------------
#    Spherical - Cylindrical
# --------------------------------------------------------

def _spherical_to_cylindrical(r,phi,theta):
    _r = r*np.sin(theta)
    z = r*np.cos(theta)
    return dict(r=_r, phi=phi, z=z)

def _cylindrical_to_spherical(r,phi,z):
    _r = np.sqrt(r*r + z*z)
    theta = np.arccos(z/_r)
    return dict(r=_r, phi=phi, theta=theta)

def spherical_to_cylindrical(r, phi=None, theta=None):

    if r is None and phi is None and theta is None:
        raise ValueError("At least one coordinate must be specified!")

    if r is not None and phi is not None and theta is None:
        # This is r-phi to x-y
        coords = _spherical_to_cylindrical(r=r, phi=phi, theta=np.pi/2.)
        del coords["z"]
        return coords

    elif r is not None and theta is not None and phi is None:
        # This is r-theta to x-z
        coords = _spherical_to_cylindrical(r=r, phi=0., theta=theta)
        del coords["phi"]
        return coords

    elif r is not None and phi is not None and theta is not None:
        return _spherical_to_cylindrical(r=r, phi=phi, theta=theta)

    else:
        raise ValueError("Unsupported operation from spherical to cylindrical coordinates.")

def cylindrical_to_spherical(r, phi=None, z=None):

    if r is None and phi is None and z is None:
        raise ValueError("At least one coordinate must be specified!")

    if r is not None and phi is not None and z is None:
        coords = _cylindrical_to_spherical(r=r, phi=phi, z=0.)
        del coords["theta"]
        return coords

    elif r is not None and z is not None and phi is None:
        coords = _cylindrical_to_spherical(r=r, phi=0., z=z)
        del coords["phi"]
        return coords

    elif r is not None and phi is not None and z is not None:
        return _cylindrical_to_spherical(r=r, phi=phi, z=z)

    else:
        raise ValueError("Unsupported operation from cylindrical to spherical coordinates.")


register_transform(SphericalCoordinates, CylindricalCoordinates, spherical_to_cylindrical, cylindrical_to_spherical)