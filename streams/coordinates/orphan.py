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

__all__ = ["OrphanCoordinates", "distance_to_orphan_plane"]

class OrphanCoordinates(coord.SphericalCoordinatesBase):
    """ A spherical coordinate system defined by the orbit of the Orphan stream
        as described in 
        http://iopscience.iop.org/0004-637X/711/1/32/pdf/apj_711_1_32.pdf

    """
    __doc__ = __doc__.format(params=coord.SphericalCoordinatesBase.\
                                          _init_docstring_param_templ.\
                                          format(lonnm='Lambda', latnm='Beta'))

    def __init__(self, *args, **kwargs):
        super(OrphanCoordinates, self).__init__()

        if len(args) == 1 and len(kwargs) == 0 and \
            isinstance(args[0], coord.SphericalCoordinatesBase):
            newcoord = args[0].transform_to(self.__class__)
            self.Lambda = newcoord.Lambda
            self.Beta = newcoord.Beta
            self._distance = newcoord._distance
        else:
            super(OrphanCoordinates, self)._initialize_latlon('Lambda', 'Beta', \
                False, args, kwargs, anglebounds=((-180, 180), (-90,90)))

    def __repr__(self):
        if self.distance is not None:
            diststr = ', Distance={0:.2g} {1!s}'.format(self.distance._value, 
                                                        self.distance._unit)
        else:
            diststr = ''

        msg = "<{0} Lambda={1:.5f} deg, Beta={2:.5f} deg{3}>"
        return msg.format(self.__class__.__name__, self.Lambda.degrees,
                          self.Beta.degrees, diststr)

    @property
    def lonangle(self):
        return self.Lambda

    @property
    def latangle(self):
        return self.Beta

# Define the Euler angles
phi = radians(128.79)
theta = radians(54.39)
psi = radians(90.70)

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z", degrees=False)
C = rotation_matrix(theta, "x", degrees=False)
B = rotation_matrix(psi, "z", degrees=False)
sgr_matrix = np.array(B.dot(C).dot(D))

# Galactic to Orphan coordinates
@transformations.transform_function(coord.GalacticCoordinates, OrphanCoordinates)
def galactic_to_orphan(galactic_coord):
    """ Compute the transformation from Galactic spherical to Orphan coordinates. 
    """

    l = galactic_coord.l.radians
    b = galactic_coord.b.radians

    X = cos(b)*cos(l)
    Y = cos(b)*sin(l)
    Z = sin(b)

    # Calculate X,Y,Z,distance in the Sgr system
    Xs, Ys, Zs = sgr_matrix.dot(np.array([X, Y, Z]))

    Zs = -Zs

    # Calculate the angular coordinates lambda,beta
    Lambda = degrees(np.arctan2(Ys,Xs))
    if Lambda<0:
        Lambda += 360

    Beta = degrees(np.arcsin(Zs/np.sqrt(Xs*Xs+Ys*Ys+Zs*Zs)))

    return OrphanCoordinates(Lambda, Beta, distance=galactic_coord.distance, 
                          unit=(u.degree, u.degree))

@transformations.transform_function(OrphanCoordinates, coord.GalacticCoordinates)
def orphan_to_galactic(orphan_coord):
    L = orphan_coord.Lambda.radians
    B = orphan_coord.Beta.radians

    Xs = cos(B)*cos(L)
    Ys = cos(B)*sin(L)
    Zs = sin(B)
    Zs = -Zs

    X, Y, Z = sgr_matrix.T.dot(np.array([Xs, Ys, Zs]))

    l = degrees(np.arctan2(Y,X))
    b = degrees(np.arcsin(Z/np.sqrt(X*X+Y*Y+Z*Z)))

    if l<0:
        l += 360

    return coord.GalacticCoordinates(l, b, distance=orphan_coord.distance, 
                                     unit=(u.degree, u.degree))

def distance_to_orphan_plane(ra, dec, heliocentric_distance):
    """ Given an RA, Dec, and Heliocentric distance, compute the distance
        to the midplane of the Orphan plane 

        Parameters
        ----------
        ra : float
            A right ascension in decimal degrees
        dec : float
            A declination in decimal degrees
        heliocentric_distance : float
            The distance from the sun to a star in kpc.

    """

    eq_coords = coord.ICRSCoordinates(ra, dec, unit=(u.degree, u.degree))
    orphan_coords = eq_coords.transform_to(OrphanCoordinates)
    orphan_coords.distance = coord.Distance(heliocentric_distance, unit=u.kpc)

    Z_orp_sol = orphan_coords.distance.kpc * np.sin(orphan_coords.Beta.radians)

    return Z_orp_sol