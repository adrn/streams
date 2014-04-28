# coding: utf-8

""" Base class for handling analytic representations of scalar gravitational
    potentials.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import copy
import inspect
import logging
import functools

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import astropy.units as u

from .. import usys

__all__ = ["Potential", "CartesianPotential", "CompositePotential", "PotentialParameter"]


class PotentialParameter(object):

    def __init__(self, value=None, truth=None, range=(), latex="", units=usys):

        if value is None and truth is None:
            raise ValueError("If value not specified, must specify truth.")

        elif value is None:
            value = truth

        if hasattr(value, "unit"):
            q = value.decompose(units)
            self._value = q.value
            self._unit = q.unit

            t = truth.decompose(units)
            self._truth = t.value

            lo,hi = range
            self._range = (lo.decompose(units).value,
                           hi.decompose(units).value)

        else:
            self._value = value
            self._truth = truth
            self._unit = u.dimensionless_unscaled
            self._range = range

        self.latex = latex

    @property
    def value(self):
        return self._value*self._unit

    @value.setter
    def value(self, v):
        self._value = v.to(self._unit).value

    @property
    def truth(self):
        return self._truth*self._unit

    @truth.setter
    def truth(self, v):
        self._truth = v.to(self._unit).value

    @property
    def range(self):
        return (self._range[0]*self._unit, self._range[1]*self._unit)

    def __float__(self):
        return self._value

class Potential(object):

    def __init__(self, units, f, f_prime, latex=None, parameters=None):
        """ A baseclass for representing gravitational potentials in Cartesian
            coordinates. You must specify the functional form of the potential
            component. You may also optionally add derivatives using the
            f_prime keyword.

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            f : function
                The functional form of the potential component. This must be a
                function that accepts N arguments where N is the dimensionality.
            f_prime : tuple (optional)
                A function that computes the derivatives of the potential.
            latex : str (optional)
                The latex representation of this potential component. Will be
                used to make sexy output in iPython Notebook.
            parameters : dict (optional)
                Any extra parameters that the potential function requires.

        """

        self.units = units

        # Convert parameters to the given units
        self.parameters = parameters
        if not parameters.has_key("r_0"):
            parameters["r_0"] = ([0.,0.,0.]*u.km).decompose(self.units)

        if not parameters["r_0"].ndim == 2:
            parameters["r_0"] = parameters["r_0"][np.newaxis]

        self._parameters = self._rescale_parameters(parameters)

        # Make sure the f is callable, and that the component doesn't already
        #   exist in the potential
        if not hasattr(f, '__call__'):
            raise TypeError("'f' parameter must be a callable function! You "
                            "passed in a '{0}'".format(f.__class__))

        self.f = lambda r: f(r, **self._parameters)

        if f_prime != None:
            if not hasattr(f_prime, '__call__'):
                raise TypeError("'f_prime' must be a callable function! You "
                                "passed in a '{0}'".format(f_prime.__class__))
            self.f_prime = lambda r: f_prime(r, **self._parameters)
        else:
            self.f_prime = None

        if latex != None:
            if latex.startswith("$"):
                latex = latex[1:]

            if latex.endswith("$"):
                latex = latex[:-1]

            self._latex = latex

    def _rescale_parameters(self, parameters):
        """ Given a dictionary of potential component parameters, trust that
            the user passed in Quantity objects where necessary. For the sake
            of speed later on, we convert any Quantity-like objects to numeric
            values in the base unit system of this potential.
        """
        _params = dict()
        for param_name, val in parameters.items():
            try:
                _params[param_name] = val.decompose(bases=self.units).value
            except AttributeError: # not Quantity-like
                _params[param_name] = val

        return _params

    def _value_at(self, r):
        """ Compute the value of the potential at the given position(s),
            assumed to be in the same system of units as the Potential.

            Parameters
            ----------
            r : ndarray
                Position to compute the value at in same units as Potential.
        """
        return self.f(r)

    def value_at(self, r):
        """ Compute the value of the potential at the given position(s)

            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the value at.
        """
        _r = r.decompose(bases=self.units).value
        c = (u.J/u.kg).decompose(bases=self.units)
        return self._value_at(_r) * u.CompositeUnit(1., c.bases, c.powers)

    def _acceleration_at(self, r):
        """ Compute the acceleration due to the potential at the given
            position(s), assumed to be in the same system of units as
            the Potential.

            Parameters
            ----------
            r : ndarray
                Position to compute the value at in same units as Potential.
        """
        return self.f_prime(r)

    def acceleration_at(self, r):
        """ Compute the acceleration due to the potential at the given
            position(s)

            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the acceleration at.
        """

        _r = r.decompose(bases=self.units).value
        c = (u.m/u.s**2).decompose(bases=self.units)
        return self._acceleration_at(_r) * u.CompositeUnit(1., c.bases,
                                                           c.powers)

    def _repr_latex_(self):
        """ Generate a latex representation of the potential. This is used by
            the IPython notebook to render nice latex equations.
        """
        return u'${0}$'.format(self._latex)


class CartesianPotential(Potential):

    def plot(self, ndim, grid, axes=None, **kwargs):
        """ Plot equipotentials lines. Must pass in grid arrays to evaluate the
            potential over (positional args). This function takes care of the
            meshgridding...

            Parameters
            ----------
            ndim : int
                Number of dimensions. TODO: this is not quite right...
            grid : astropy.units.Quantity
                Coordinate grid to compute the potential on. Should be a 1D
                array, and is used for all dimensions.
            axes : matplotlib.Axes (optional)
            kwargs : dict
                kwargs passed to either contourf() or plot().

        """

        if not hasattr(grid, 'unit'):
            raise TypeError("grid must be a Quantity object")

        if axes == None:
            if ndim > 1:
                fig, axes = plt.subplots(ndim-1, ndim-1,
                                         sharex=True, sharey=True,
                                         figsize=(12,12))
            else:
                fig, axes = plt.subplots(1, 1, figsize=(12,12))

        try:
            axes[0,0]
        except TypeError:
            axes = np.array([[axes]])

        fig = axes[0,0].figure

        if ndim == 1:
            raise NotImplementedError("1D potential not implemented")
            axes[0,0].plot(grid, self.value_at(grid))
            axes[0,0].set_xlabel("[{0}]".format(grid.unit))
            axes[0,0].set_ylabel(self._repr_latex_())
            return fig,axes

        else:
            for i in range(1,ndim):
                for jj in range(ndim-1):
                    ii = i-1
                    if jj > ii:
                        axes[ii,jj].set_visible(False)
                        continue

                    X1, X2 = np.meshgrid(grid.value,grid.value)

                    r = np.array([np.zeros_like(X1.ravel()).tolist() \
                                    for xx in range(ndim)])
                    r[jj] = X1.ravel()
                    r[i] = X2.ravel()

                    Z = self._value_at(r.T).reshape(X1.shape)
                    cs = axes[ii,jj].contourf(X1, X2, Z, cmap=cm.bone_r, **kwargs)

            cax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
            fig.colorbar(cs, cax=cax)

            # Label the axes
            for jj in range(ndim-1):
                axes[-1,jj].set_xlabel("[{0}]".format(grid.unit))
                axes[jj,0].set_ylabel("[{0}]".format(grid.unit))

        fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.08, bottom=0.08, top=0.9, right=0.9 )
        #fig.suptitle(self._repr_latex_(), fontsize=24)

        return fig, axes

class CompositePotential(dict, CartesianPotential):

    def __init__(self, units, *args, **kwargs):
        """ Represents a potential composed of several sub-potentials. For
            example, two point masses or a galactic disk + halo. The origins
            of the components are *relative to the origin of the composite*.

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
        """
        self.units = units

        for v in kwargs.values():
            if not isinstance(v, Potential):
                raise TypeError("Values may only be Potential objects, not "
                                "{0}.".format(type(v)))

        dict.__init__(self, *args, **kwargs)

    def __repr__(self):
        return "<CompositePotential: {0}>".format(",".join(self.keys()))

    def __setitem__(self, key, value):
        if not isinstance(value, Potential):
            raise TypeError("Values may only be Potential objects, not "
                            "{0}.".format(type(value)))

        super(CompositePotential, self).__setitem__(key, value)

    @property
    def _latex(self):
        return "$\n$".join(set([x._latex for x in self.values()]))

    def _value_at(self, r):
        """ Compute the value of the potential at the given position(s),
            assumed to be in the same system of units as the Potential.

            Parameters
            ----------
            r : ndarray
                Position to compute the value at in same units as Potential.
        """
        return np.sum([p._value_at(r) for p in self.values()], axis=0)

    def _acceleration_at(self, r):
        """ Compute the acceleration due to the potential at the given
            position(s), assumed to be in the same system of units as
            the Potential.

            Parameters
            ----------
            r : ndarray
                Position to compute the value at in same units as Potential.
        """
        return np.sum([p._acceleration_at(r) for p in self.values()], axis=0)