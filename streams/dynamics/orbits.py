# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G
import triangle

from .core import _validate_quantity, DynamicalBase
from .particles import Particle

__all__ = ["Orbit"]

class Orbit(DynamicalBase):

    def __init__(self, t, r, v, m=None, units=None):
        """ Represents a collection of orbits, e.g. positions and velocities
            over time for a set of particles.

            Input r and v should be shape (ntimesteps, nparticles, ndim).

            Parameters
            ----------
            t : astropy.units.Quantity
                Time.
            r : astropy.units.Quantity
                Position of the particle(s). Should have shape
                (ntimesteps, nparticles, ndim).
            v : astropy.units.Quantity
                Velocity of the particle(s). Should have shape
                (ntimesteps, nparticles, ndim).
            m : astropy.units.Quantity (optional)
                Mass.
            units : list (optional)
                A list of units defining the desired unit system for
                the particles. If not provided, will use the units of
                the input Quantities to define a system of units. Mainly
                used for internal representations.
        """

        _validate_quantity(t, unit_like=u.s)
        _validate_quantity(r, unit_like=u.km)
        _validate_quantity(v, unit_like=u.km/u.s)

        try:
            self.ntimesteps, self.nparticles, self.ndim = r.value.shape
        except ValueError:
            raise ValueError("Position and velocity should have shape "
                             "(ntimesteps, nparticles, ndim)")

        if units is None and m is None:
            raise ValueError("If not specifying list of units, you must "
                             "specify a mass Quantity for the particles.")
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

        if len(t.value) != self.ntimesteps:
            raise ValueError("Length of time array must match number of "
                             "timesteps in position and velocity.")

        # decompose each input into the specified unit system
        _r = r.decompose(self.units).value
        _v = v.decompose(self.units).value

        # create container for all 6 phasespace
        self._X = np.zeros((self.ntimesteps, self.nparticles, self.ndim*2))
        self._X[..., :self.ndim] = _r
        self._X[..., self.ndim:] = _v
        self._m = m.decompose(self.units).value
        self._t = t.decompose(self.units).value

    @property
    def t(self):
        t_unit = filter(lambda x: x.is_equivalent(u.s), self.units)[0]
        return self._t * t_unit

    def to(self, units):
        """ Return a new Orbit in the specified unit system. """
        return Orbit(t=self.t, r=self.r, v=self.v, m=self.m, units=units)

    def plot_r(self, coord_names=['x','y','z'], **kwargs):
        """ Make a scatter-plot of 3 projections of the orbit positions.

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

        fig = triangle.corner(np.vstack(self._r), labels=labels,
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

        fig = triangle.corner(np.vstack(self._v), labels=labels,
                              plot_contours=False, plot_datapoints=True,
                              **kwargs)
        return fig

    def __getitem__(self, key):
        """ Slice on time """

        if isinstance(key, slice) :
            #Get the start, stop, and step from the slice
            return Orbit(t=self.t[key], r=self.r[key],
                         v=self.v[key], m=self.m[key], units=self.units)

        elif isinstance(key, int) :
            return Particle(r=self.r[key], v=self.v[key], m=self.m[key],
                            units=self.units)

        else:
            raise TypeError("Invalid argument type.")
