# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
from collections import defaultdict

# Third-party
import numpy as np
import astropy.units as u

# project
from ..coordinates import _gc_to_hel, _hel_to_gc

# Create logger
logger = logging.getLogger(__name__)

_transform_graph = dict()
_transform_graph["heliocentric"]["galactocentric"] = _hel_to_gc
_transform_graph["galactocentric"]["heliocentric"] = _gc_to_hel

class ReferenceFrame(object):

    def __init__(self, name, coord_names):
        self.name = name
        self.coord_names = coord_names
        self.ndim = len(self.coord_names)

    def __repr__(self):
        return "<Frame: {}, {}D>".format(self.name, self.ndim)

    def to(self, other, X):
        """ Transform the coordinates X to the reference frame 'other'.
            X should have units of the global usys.

            Parameters
            ----------
            other : ReferenceFrame
            X : array_like
        """
        new_X = _transform_graph[self.name][other.name](X)
        return new_X

heliocentric = ReferenceFrame(name="heliocentric",
                              coord_names=("l","b","D","mul","mub","vr"))

galactocentric = ReferenceFrame(name="galactocentric",
                                coord_names=("x","y","z","vx","vy","vz"))

'''
def cartesian_to_spherical_position(x,y,z):
    pass

def spherical_to_cartesian_position(r,lon,lat):
    pass

def cartesian_to_spherical_velocity(x,y,z):
    pass

def spherical_to_cartesian_velocity(r,lon,lat):
    pass

transform_graph = defaultdict(defaultdict(dict))

transform_graph[Cartesian][Spherical]["position"] = \
    cartesian_to_spherical_position
transform_graph[Cartesian][Spherical]["velocity"] = \
    cartesian_to_spherical_velocity

transform_graph[Spherical][Cartesian]["position"] = \
    spherical_to_cartesian_position
transform_graph[Spherical][Cartesian]["velocty"] = \
    spherical_to_cartesian_velocity

class CoordinateSystem(object):

    def __init__(self, *args):
        self._coords = args

    def to(self, Other):
        """ Transform to the other coordinate system """

        if not isinstance(other, type):
            raise ValueError("Other system must also be a CoordinateSystem "
                             "subclass.")

        this_class = self.__class__
        phys_type = self.physical_type
        transform_func = transform_graph[this_class][Other][phys_type]
        new_coords = transform_func(*self._coords)

        return Other(*new_coords)

class Cartesian(CoordinateSystem):

    def __init__(self, x, y, z, physical_type="position"):
        self.physical_type = physical_type
        super(Cartesian, self).__init__(x, y, z)

class Spherical(CoordinateSystem):

    def __init__(self, r, lon, lat, physical_type="position"):
        self.physical_type = physical_type
        super(Spherical, self).__init__(x, y, z)

# TODO: add support for cylindrical
# class Cylindrical(object):
#    pass


class Heliocentric(object):

    def __init__(self, coords):
        pass

Heliocentric(Spherical(lon=..., lat=..., distance=...))
'''