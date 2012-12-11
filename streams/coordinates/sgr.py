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

__all__ = ["SgrCoordinates"]

@transformations.coordinate_alias('sgr')
class SgrCoordinates(coord.SphericalCoordinatesBase):
    """
    A spherical coordinate system defined by the orbit of the Sagittarius dwarf galaxy, as described in
    http://adsabs.harvard.edu/abs/2003ApJ...599.1082M and further explained here: http://www.astro.virginia.edu/~srm4n/Sgr/.

    """
    __doc__ = __doc__.format(params=coord.SphericalCoordinatesBase._init_docstring_param_templ.format(lonnm='Lambda', latnm='Beta'))

    def __init__(self, *args, **kwargs):
        super(SgrCoordinates, self).__init__()

        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], coord.SphericalCoordinatesBase):
            newcoord = args[0].transform_to(self.__class__)
            self.Lambda = newcoord.Lambda
            self.Beta = newcoord.Beta
            self._distance = newcoord._distance
        else:
            super(SgrCoordinates, self)._initialize_latlon('Lambda', 'Beta', False, args, kwargs)

    def __repr__(self):
        if self.distance is not None:
            diststr = ', Distance={0:.2g} {1!s}'.format(self.distance._value, self.distance._unit)
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

# Galactic to Sgr coordinates
@transformations.static_transform_matrix(coord.GalacticCoordinates, SgrCoordinates)
def galactic_to_sgr():
    """ Compute the rotation matrix to transform from Spherical Galactic Coordinates to
        the Sgr coordinate system.
    """

    phi = radians(180+3.75)
    theta = radians(90-13.46)
    psi = radians(180+14.111534)

    rot11 = cos(psi)*cos(phi)-cos(theta)*sin(phi)*sin(psi)
    rot12 = cos(psi)*sin(phi)+cos(theta)*cos(phi)*sin(psi)
    rot13 = sin(psi)*sin(theta)
    rot21 = -sin(psi)*cos(phi)-cos(theta)*sin(phi)*cos(psi)
    rot22 = -sin(psi)*sin(phi)+cos(theta)*cos(phi)*cos(psi)
    rot23 = cos(psi)*sin(theta)
    rot31 = sin(theta)*sin(phi)
    rot32 = -sin(theta)*cos(phi)
    rot33 = cos(theta)

    m = np.array([[rot11, rot12, rot13], [rot21, rot22, rot23], [rot31, rot32, rot33]])

    return np.asmatrix(m)

@transformations.static_transform_matrix(SgrCoordinates, coord.GalacticCoordinates)
def sgr_to_galactic():
    return galactic_to_sgr().T