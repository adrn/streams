# coding: utf-8

""" Base class for Particle and Orbit """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy

# Third-party
import numpy as np
import astropy.units as u
import triangle

# Project
from .. import usys
from ..coordinates.frame import ReferenceFrame

class DynamicalBase(object):

    def __init__(self, coords, frame, units=None, meta=dict()):
        """ Represents a dynamical particle or collection of particles.
            Particles can have associated metadata, e.g., mass or for
            a Satellite, velocity dispersion.

            Parameters
            ----------
            coords : iterable
                Can either be an iterable (e.g., list or tuple) of Quantity
                objects, in which case their units are grabbed from the
                objects themselves, or an array_like object with the first
                axis as the separate coordinate dimensions, and the units
                parameter specifying the units along each dimension.
            frame : ReferenceFrame
                The reference frame that the particles are in.
            units : iterable (optional)
                Must be specified if q is an array_like object, otherwise this
                is constructed from the Quantity objects in q.
            meta : dict (optional)
                Any additional metadata.
        """

        try:
            self.ndim = coords.shape[0]
        except AttributeError:
            self.ndim = len(coords)

        if frame.ndim != self.ndim:
            raise ValueError("ReferenceFrame must have same dimensions as coordinates "
                             "({} vs. {})".format(frame.ndim, self.ndim))

        # now build an internal contiguous numpy array to store the coordinate
        #   data in the system 'usys'
        self._repr_units = []
        for ii in range(self.ndim):
            # make sure this dimension's data is at least 1D
            q = np.atleast_1d(coords[ii]).copy()

            if hasattr(q, "unit") and q.unit != u.dimensionless_unscaled:
                unit = q.unit
                value = q.decompose(usys).value
            else:
                try:
                    unit = units[ii]
                except TypeError:
                    raise ValueError("Must specify units for each coordinate dimension "
                                     "if the data are not Quantities.")
                value = (q*unit).decompose(usys).value

            # preserve the input units to display with
            self._repr_units.append(unit)

            try:
                self._X[...,ii] = value
            except AttributeError:
                self._X = np.zeros(q.shape + (self.ndim,))
                self._X[...,ii] = value

        self.nparticles = self._X.shape[-2]

        # validate reference frame
        if not isinstance(frame, ReferenceFrame):
            raise TypeError("frame must be a valid ReferenceFrame object.")
        self.frame = frame

        # find units in usys that match the physical types of each units
        self._internal_units = []
        for unit in self._repr_units:
            self._internal_units.append((1*unit).decompose(usys).unit)

        self.meta = meta.copy()
        for k,v in self.meta.items():
            setattr(self,k,v)

    def __repr__(self):
        return "<{0}: ndim={1}, frame={2}>".format(self.__class__, self.ndim, self.frame)

    def copy(self):
        """ Return a copy of the current instance """
        return copy.deepcopy(self)

    def __getitem__(self, slc):
        if isinstance(slc, (int,slice)):
            raise ValueError("Slicing not supported by single index or slice.")

        else:
            try:
                ii = self.frame.coord_names.index(slc)
            except ValueError:
                raise AttributeError("Invalid coordinate name {}".format(slc))

            return (self._X[...,ii]*self._internal_units[ii]).to(self._repr_units[ii])

    @property
    def _repr_X(self):
        """ Return the 6D array of all coordinates in the repr units """

        _repr_X = np.zeros_like(self._X)
        for ii in range(self.ndim):
            _repr_X[...,ii] = (self._X[...,ii]*self._internal_units[ii])\
                               .to(self._repr_units[ii]).value

        return _repr_X
