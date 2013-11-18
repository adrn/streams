# coding: utf-8

""" Astropy coordinate class for the Sagittarius coordinate system """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from numpy import radians, degrees, cos, sin

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import transformations
from astropy.coordinates.angles import rotation_matrix

__all__ = ["OrphanCoordinates", "distance_to_orp_plane"]

class OrphanCoordinates(coord.SphericalCoordinatesBase):
    """ A spherical coordinate system defined by the orbit of the Orphan
        stream as described in
        http://iopscience.iop.org/0004-637X/711/1/32/pdf/apj_711_1_32.pdf

    """
    __doc__ = __doc__.format(params=coord.SphericalCoordinatesBase. \
            _init_docstring_param_templ.format(lonnm='Lambda', latnm='Beta'))

    def __init__(self, *args, **kwargs):
        super(OrphanCoordinates, self).__init__()

        if len(args) == 1 and len(kwargs) == 0 and \
            isinstance(args[0], coord.SphericalCoordinatesBase):

            newcoord = args[0].transform_to(self.__class__)
            self._lonangle = newcoord._lonangle
            self._latangle = newcoord._latangle
            self._distance = newcoord._distance
        else:
            super(OrphanCoordinates, self).\
                _initialize_latlon('Lambda', 'Beta', args, kwargs)

    #strings used for making __repr__ work
    _repr_lon_name = 'Lambda'
    _repr_lat_name = 'Beta'

    # Default format for to_string
    _default_string_style = 'dmsdms'

    @property
    def Lambda(self):
        return self._lonangle

    @property
    def Beta(self):
        return self._latangle

# Define the Euler angles
phi = radians(128.79)
theta = radians(54.39)
psi = radians(90.70)

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z", unit=u.radian)
C = rotation_matrix(theta, "x", unit=u.radian)
B = rotation_matrix(psi, "z", unit=u.radian)
sgr_matrix = np.array(B.dot(C).dot(D))

# Galactic to Orphan coordinates
@transformations.transform_function(coord.Galactic, OrphanCoordinates)
def galactic_to_orphan(galactic_coord):
    """ Compute the transformation from Galactic spherical to Orphan coordinates.
    """

    l = galactic_coord.l.radian
    b = galactic_coord.b.radian

    X = cos(b)*cos(l)
    Y = cos(b)*sin(l)
    Z = sin(b)

    # Calculate X,Y,Z,distance in the Orp system
    Xs, Ys, Zs = sgr_matrix.dot(np.array([X, Y, Z]))

    # Calculate the angular coordinates lambda,beta
    Lambda = degrees(np.arctan2(Ys,Xs))
    if Lambda<0:
        Lambda += 360

    Beta = degrees(np.arcsin(Zs/np.sqrt(Xs*Xs+Ys*Ys+Zs*Zs)))

    return OrphanCoordinates(Lambda, Beta, distance=galactic_coord.distance,
                          unit=(u.degree, u.degree))

@transformations.transform_function(OrphanCoordinates, coord.Galactic)
def orphan_to_galactic(orphan_coord):
    L = orphan_coord.Lambda.radian
    B = orphan_coord.Beta.radian

    Xs = cos(B)*cos(L)
    Ys = cos(B)*sin(L)
    Zs = sin(B)

    X, Y, Z = sgr_matrix.T.dot(np.array([Xs, Ys, Zs]))

    l = degrees(np.arctan2(Y,X))
    b = degrees(np.arcsin(Z/np.sqrt(X*X+Y*Y+Z*Z)))

    if l<0:
        l += 360

    return coord.GalacticCoordinates(l, b, distance=orphan_coord.distance,
                                     unit=(u.degree, u.degree))

def distance_to_orp_plane(ra, dec, heliocentric_distance):
    """ Given an RA, Dec, and Heliocentric distance, compute the distance
        to the midplane of the Orphan orbital plane

        Parameters
        ----------
        ra : astropy.coordinates.Angle
            Right ascension.
        dec : astropy.coordinates.Angle
            Declination.
        heliocentric_distance : Quantity
            The distance from the Sun to a star.

    """

    icrs = coord.ICRSCoordinates(ra, dec)
    sgr_coords = icrs.transform_to(OrphanCoordinates)
    sgr_coords.distance = coord.Distance(heliocentric_distance)

    Z_sgr = sgr_coords.distance * np.sin(sgr_coords.Beta.radian)
    return Z_sgr