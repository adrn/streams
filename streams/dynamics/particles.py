# coding: utf-8

""" Massive particles or collections of particles """

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
from ..coordinates import _gc_to_hel, _hel_to_gc

__all__ = ["Particle"]

class Particle(object):

    def __init__(self, coords, names, units=None, meta=dict()):
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
            names : iterable
                Names of each coordinate dimension. These end up being
                attributes of the object.
            units : iterable (optional)
                Must be specified if q is an array_like object, otherwise this
                is constructed from the Quantity objects in q.
            meta : dict (optional)
                Any additional metadata.
        """

        self.ndim = len(coords)

        _X = None
        _repr_units = []
        for ii in range(self.ndim):
            q = coords[ii]

            if _X is None:
                _X = np.zeros((self.ndim,) + q.shape)

            if hasattr(q, "unit") and q.unit != u.dimensionless_unscaled:
                unit = q.unit
                value = q.decompose(usys).value
            else:
                try:
                    unit = units[ii]
                except TypeError:
                    raise ValueError("Must specify units for each"
                                     "coordinate dimension.")
                value = (q*unit).decompose(usys).value

            _repr_units.append(unit)
            _X[ii] = value

        self._repr_units = _repr_units
        self._X = _X

        #if self._X.ndim > 2:
        #    raise ValueError("Particle coordinates must be 1D.")

        # find units in usys that match the physical types of each units
        self._internal_units = []
        for unit in self._repr_units:
            self._internal_units.append((1*unit).decompose(usys).unit)

        if len(names) != self.ndim:
            raise ValueError("Must specify coordinate name for each "
                             "dimension.")
        self.names = names
        self.meta = meta

    def copy(self):
        """ Return a copy of the current instance. I'm just a copy
            of a copy of a copy...
        """
        return copy.deepcopy(self)

    def __getitem__(self, slc):
        if isinstance(slc, (int,slice)):
            cpy = self.copy()
            cpy._X = cpy._X[:,slc]
            return cpy

        else:
            try:
                ii = self.names.index(slc)
            except ValueError:
                raise AttributeError("Invalid coordinate name {}".format(slc))

            return (self._X[ii]*self._internal_units[ii])\
                        .to(self._repr_units[ii])

    @property
    def _repr_X(self):
        """ Return the 6D array of all coordinates in the repr units """

        _repr_X = []
        for ii in range(self.ndim):
            _repr_X.append((self._X[ii]*self._internal_units[ii])\
                               .to(self._repr_units[ii]).value.tolist())

        return np.array(_repr_X)

    @property
    def nparticles(self):
        return self._X.shape[1]

    def plot(self, fig=None, labels=None, **kwargs):
        """ Make a corner plot showing all dimensions.

        """

        if labels is None:
            labels = ["{0} [{1}]".format(n,uu) \
                        for n,uu in zip(self.names, self._repr_units)]

        if fig is not None:
            kwargs["fig"] = fig

        fig = triangle.corner(self._repr_X.T,
                              labels=labels,
                              plot_contours=False,
                              plot_datapoints=True,
                              **kwargs)

        return fig

    def decompose(self, units):
        """ Decompose each coordinate axis to the given unit system """

        q = [self[n].decompose(units) for n in self.names]
        return Particle(q, self.names, meta=self.meta)

'''
def observe(self, error_model):
        """ Given an error model, transform to heliocentric coordinates,
            apply errors models, transform back and return a new Particle
            object.
        """
        _X = self._X[:]
        hel = _gc_to_hel(_X)
        hel_err = error_model(hel)

        O = np.random.normal(hel, hel_err) # observed
        return O, hel_err

'''