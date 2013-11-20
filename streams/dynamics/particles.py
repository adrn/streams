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
from astropy.constants import G
import triangle

# Project
from .. import usys
from .core import _validate_quantity, DynamicalBase
from ..coordinates import _gc_to_hel, _hel_to_gc

__all__ = ["Particle"]


class Particle(object):

    def __init__(self, coords=(), names=(), units=None, meta=dict()):
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
            meta : dict (optoonal)
                Any additional metadata.
        """

        self.ndim = len(coords)

        _X = None
        _repr_units = []
        for ii in range(self.ndim):
            q = coords[ii]

            if _X is None:
                _X = np.zeros((self.ndim,) + q.shape)

            if hasattr(q, "unit"):
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

        if self._X.ndim > 2:
            raise ValueError("Particle coordinates must be 1D.")

        # find units in usys that match the physical types of each units
        self._internal_units = []
        for unit in self._repr_units:
            self._internal_units.append((1*unit).decompose(usys).unit)

        if len(names) != self.ndim:
            raise ValueError("Must specify coordinate name for each "
                             "dimension.")
        self.names = names

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

    def __len__(self):
        return self._X.shape[1]

    def plot(self, fig=None, labels=None, **kwargs):
        """ Make a corner plot showing all dimensions.

        """

        if labels is None:
            labels = ["{0} [{1}]".format(n,uu) \
                        for n,uu in zip(self.names,self._repr_units)]

        if fig is not None:
            kwargs["fig"] = fig

        fig = triangle.corner(self._repr_X.T,
                              labels=labels,
                              plot_contours=False,
                              plot_datapoints=True,
                              **kwargs)

        return fig

class OldParticle(DynamicalBase):

    def __init__(self, r, v, m=None, units=None):
        """ A represents a dynamical particle or collection of particles.
            Particles can have mass or be massless.

            Parameters
            ----------
            r : astropy.units.Quantity
                Position of the particle(s). Should have shape
                (nparticles, ndim).
            v : astropy.units.Quantity
                Velocity of the particle(s). Should have shape
                (nparticles, ndim).
            m : astropy.units.Quantity (optional)
                Mass of the particle(s). Should have shape (nparticles, ).
            units : list (optional)
                A list of units defining the desired unit system for
                the particles. If not provided, will use the units of
                the input Quantities to define a system of units. Mainly
                used for internal representations.
        """

        # Make sure position has position-like units, same for velocity
        _validate_quantity(r, unit_like=u.km)
        _validate_quantity(v, unit_like=u.km/u.s)

        if r.value.ndim == 1:
            r = r.reshape((1, len(r)))
            v = v.reshape((1, len(v)))

        try:
            self.nparticles, self.ndim = r.value.shape
        except ValueError:
            raise ValueError("Position and velocity should have shape "
                             "(nparticles, ndim)")

        if units is None and m is None:
            raise ValueError("If not specifying a list of units, you must "
                             "specify a mass Quantity for the particles to"
                             "complete the unit system specification.")
        elif units is not None and m is None:
            m = ([0.] * self.nparticles * u.kg).decompose(units)

        _validate_quantity(m, unit_like=u.kg)

        if units is None:
            _units = [r.unit, m.unit] + v.unit.bases + [u.radian]
            self.units = set(_units)
        else:
            self.units = units

        unq_types = np.unique([x.physical_type for x in self.units])
        if len(self.units) != len(unq_types):
            raise ValueError("Multiple units specify the same physical type!")

        if r.value.shape != v.value.shape:
            raise ValueError("Position and velocity must have same shape.")

        # decompose each input into the specified unit system
        _r = r.decompose(self.units).value
        _v = v.decompose(self.units).value

        # create container for all 6 phasespace
        self._X = np.zeros((self.nparticles, self.ndim*2))
        self._X[:,:self.ndim] = _r
        self._X[:,self.ndim:] = _v
        self._m = m.decompose(self.units).value

        # Create internal G in the correct unit system for speedy acceleration
        #   computation
        self._G = G.decompose(self.units).value

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

    def to(self, units):
        """ Return a new Particle in the specified unit system. """

        return Particle(r=self.r, v=self.v, m=self.m, units=units)

    def copy(self):
        return copy.deepcopy(self)

    def plot_r(self, coord_names=['x','y','z'], **kwargs):
        """ Make a scatter-plot of 3 projections of the positions of the
            particle coordinates.

            Parameters
            ----------
            coord_names : list
                Name of each axis, e.g. ['x','y','z']
            kwargs (optional)
                Keyword arguments that get passed to triangle.corner()
        """
        from ..plot.data import scatter_plot_matrix
        if not len(coord_names) == self.ndim:
            raise ValueError("Must pass a coordinate name for each dimension.")

        labels = [r"{0} [{1}]".format(nm, self.r.unit)
                    for nm in coord_names]

        kwargs["ms"] = kwargs.get("ms", 2.)
        kwargs["alpha"] = kwargs.get("alpha", 0.5)

        fig = triangle.corner(self._r, labels=labels,
                              plot_contours=False, plot_datapoints=True,
                              **kwargs)
        return fig

    def plot_v(self, coord_names=['vx','vy','vz'], **kwargs):
        """ Make a scatter-plot of 3 projections of the velocities of the
            particle coordinates.

            Parameters
            ----------
            coord_names : list
                Name of each axis, e.g. ['Vx','Vy','Vz']
            kwargs (optional)
                Keyword arguments that get passed to triangle.corner()
        """
        from ..plot.data import scatter_plot_matrix
        assert len(coord_names) == self.ndim, "Must pass a coordinate name for each dimension."

        labels = [r"{0} [{1}]".format(nm, self.v.unit)
                    for nm in coord_names]

        kwargs["ms"] = kwargs.get("ms", 2.)
        kwargs["alpha"] = kwargs.get("alpha", 0.5)

        fig = triangle.corner(self._v, labels=labels,
                              plot_contours=False, plot_datapoints=True,
                              **kwargs)
        return fig

    def merge(self, other):
        """ Merge two particle collections. Takes unit system from the first
            Particle object.
        """

        if not isinstance(other, Particle):
            raise TypeError("Can only merge two Particle objects!")

        other_r = other.r.decompose(self.units).value
        other_v = other.v.decompose(self.units).value
        other_m = other.m.decompose(self.units).value

        r = np.vstack((self._r,other_r)) * self.r.unit
        v = np.vstack((self._v,other_v)) * self.v.unit
        m = np.append(self._m,other_m) * self.m.unit

        return Particle(r=r, v=v, m=m, units=self.units)

    def __getitem__(self, key):
        r = self.r[key]
        v = self.v[key]
        m = self.m[key]

        return Particle(r=r, v=v, m=m, units=self.units)

    def __len__(self):
        return self.nparticles
