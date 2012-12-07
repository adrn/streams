# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import abc
import os, sys
import inspect

# Third-party
import numpy as np
from astropy.utils.misc import isiterable

__all__ = ["Coordinates", "CartesianCoordinates", "CylindricalCoordinates", "SphericalCoordinates", "register_transform"]

# Notes
# -----
#    - Angular coordinates assume radians, unless you provide an Angle object
#    -

def _coords_property_factory(self, key):
    """ To get around strange lambda scope issues...
        See: http://stackoverflow.com/questions/938429/scope-of-python-lambda-functions-and-their-parameters
    """
    return lambda self: self._coords[key]

class Coordinates(object):
    __metaclass__ = abc.ABCMeta

    def __new__(cls, *args, **kwargs):
        if cls is Coordinates:
            raise TypeError("Coordinates class may not be instantiated -- use a subclass defining a specific coordinate system.")
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, **kwargs):
        """ Any subclasses must overwrite the _transforms property! """
        self._coords = kwargs

        for key in self._coords.keys():
            setattr(self.__class__, key, property(_coords_property_factory(self, key)))

    def __repr__(self):
        return "<Coordinates: " + ", ".join(["{0}={1}".format(k,v) for k,v in self._coords.items()]) + ">"

    def __str__(self):
        return self.__repr__()

    def to(self, other):
        """ Given another Coordinates subclass, transform this coordinate system to the other and return
            a new Coordinates object in the new coordinate system.
        """

        if isinstance(other, Coordinates):
            raise TypeError("You must pass a Coordinates-like class, not an instance, to specify the target coordinate system (e.g. CartesianCoordinates).")

        if not inspect.isclass(other):
            raise TypeError("You must pass a Coordinates-like class to specify the target coordinate system (e.g. CartesianCoordinates).")

        if other not in self._transforms.keys():
            raise ValueError("Transformations not supported from {0} to {1}. To register a new transformation, use the register_transform() function.".format(self, other))

        new_coords = self._transforms[other](**self._coords)
        return other(**new_coords)

class CartesianCoordinates(Coordinates):
    _transforms = dict()
    _axis_names = ["X", "Y", "Z"]

    def __repr__(self):
        return "<CartesianCoordinates: " + ", ".join(["{0}={1}".format(k,v) for k,v in self._coords.items()]) + ">"


class SphericalCoordinates(Coordinates):
    _transforms = dict()
    _axis_names = ["r", "phi", "theta"]

    def __repr__(self):
        return "<SphericalCoordinates: " + ", ".join(["{0}={1}".format(k,v) for k,v in self._coords.items()]) + ">"

class CylindricalCoordinates(Coordinates):
    _transforms = dict()
    _axis_names = ["R", "phi", "Z"]

    def __repr__(self):
        return "<CylindricalCoordinates: " + ", ".join(["{0}={1}".format(k,v) for k,v in self._coords.items()]) + ">"

def register_transform(system1, system2, func12, func21):
    ''' Register a transformation from on coordinate system to another, and back again. '''

    system1._transforms[system2] = func12
    system2._transforms[system1] = func21