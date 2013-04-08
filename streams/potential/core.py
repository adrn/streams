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
from astropy.utils import isiterable
import astropy.units as u

#from ..util import *

__all__ = ["CartesianPotential"]

class CartesianPotential(object):

    def __init__(self, unit_bases):
        """ A baseclass for representing gravitational potentials in Cartesian
            coordinates.
        """
        self.unit_bases = unit_bases
    
        # Initialize empty containers for potential components and their
        #   derivatives.
        self._potential_components = dict()
        self._potential_component_derivs = dict()
        self._latex = dict()
        self.ndim = None
        self.parameters = dict()

    def add_component(self, name, func, f_prime=None, latex=None, parameters=None):
        """ Add a component to the potential. The component must have a name,
            and you must specify the functional form of the potential component.
            You may also optionally add derivatives using the 'derivs'
            keyword.

            Parameters
            ----------
            name : str, hashable
                The name of the potential component, e.g. 'halo'
            func : function
                The functional form of the potential component. This must be a
                function that accepts N arguments where N is the dimensionality.
            f_prime : tuple (optional)
                A functions that computes the derivatives of the potential.
            latex : str (optional)
                The latex representation of this potential component. Will be
                used to make sexy output in iPython Notebook.
            parameters : dict (optional)
                Any extra parameters that the potential function requires.

        """

        # Make sure the func is callable, and that the component doesn't already
        #   exist in the potential
        if not hasattr(func, '__call__'):
            raise TypeError("'func' parameter must be a callable function! You "
                            "passed in a '{0}'".format(func.__class__))

        if self._potential_components.has_key(name):
            raise NameError("Potential component '{0}' already exists!".\
                             format(name))

        if isinstance(func, functools.partial):
            func = func.func
            
        ndim_this_func = len(inspect.getargspec(func).args) - len(parameters)
        if self.ndim == None:
            self.ndim = ndim_this_func
        elif ndim_this_func != self.ndim:
            raise ValueError("This potential is already established to be "
                             "{0} dimensional. You attempted to add a component"
                             " with only {1} dimensions".\
                             format(self.ndim, ndim_this_func))
        
        self._potential_components[name] = functools.partial(func, **parameters)
        
        # If the user passes the potential derivatives
        if f_prime is not None:
            self._potential_component_derivs[name] = functools.partial(f_prime, **parameters)
        else:
            self._potential_component_derivs[name] = None

        if latex != None:
            if latex.startswith("$"):
                latex = latex[1:]

            if latex.endswith("$"):
                latex = latex[:-1]

            self._latex[name] = latex
        
        self.parameters[name] = parameters
        
    def value_at(self, r):
        """ Compute the value of the potential at the given position(s) 
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the value at.
        """
        
        x,y,z = self._r_to_xyz(r)
        
        # Compute contribution to the potential from each component
        for potential_component in self._potential_components.values():
            try:
                potential_value += potential_component(x,y,z)
            except NameError:
                potential_value = potential_component(x,y,z)
        
        return potential_value

    def acceleration_at(self, r):
        """ Compute the acceleration due to the potential at the given 
            position(s) 
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the acceleration at.
        """
        
        x,y,z = self._r_to_xyz(r)

        for potential_derivative in self._potential_component_derivs.values():
            try:
                acceleration -= potential_derivative(x,y,z)
            except NameError:
                acceleration = -potential_derivative(x,y,z)

        return acceleration

    def _repr_latex_(self):
        ''' Generate a latex representation of the potential. This is used by
            the IPython notebook to render nice latex equations.
        '''

        ltx_str = ""
        for name,latex in self._latex.items():
            ltx_str += "\\textit{{{0}}}: {1}\\\\".format(name, latex)

        if ltx_str == "":
            return ""

        return u'Components: ${0}$'.format(ltx_str)

    def _r_to_xyz(self, r):
        if isinstance(r, u.Quantity):
            r = r.decompose(bases=self.unit_bases).value
        
        if len(r.shape) == 1: 
            x,y,z = r
        else:
            x = r[:,0]
            y = r[:,1]
            z = r[:,2]
        
        return x,y,z

    def plot(self, x, y, z, axes=None, plot_kwargs=dict()):
        """ Plot equipotentials lines. Must pass in grid arrays to evaluate the
            potential over (positional args). This function takes care of the
            meshgridding...
            
            Parameters
            ----------
            x,y,z : astropy.units.Quantity
                Coordinate grids to compute the potential over.
            axes : matplotlib.Axes (optional)
            plot_kwargs : dict
                kwargs passed to either contourf() or plot().

        """
        
        coords = [x,y,z]
        
        assert x.unit == y.unit
        assert z.unit == z.unit
        
        if axes == None:
            if self.ndim > 1:
                fig, axes = plt.subplots(self.ndim-1, self.ndim-1, sharex=True, sharey=True, figsize=(12,12))
            else:
                fig, axes = plt.subplots(1, 1, figsize=(12,12))
        else:
            if self.ndim > 1:
                fig = axes[0,0].figure
            else:
                fig = axes.figure

        if self.ndim > 2:
            for ii in range(self.ndim):
                for jj in range(self.ndim):
                    if jj > ii or jj == 2:
                        try:
                            axes[ii,jj].set_visible(False)
                        except:
                            pass
                        continue

                    bottom = coords[jj]
                    side = coords[ii]
                    X1, X2 = np.meshgrid(bottom,side)

                    r = np.array([np.zeros_like(X1.ravel()) for xx in range(self.ndim)])
                    r[jj] = X1.ravel()
                    r[ii] = X2.ravel()
                    r = r.T*x.unit
                    
                    cs = axes[ii-1,jj].contourf(X1, X2, 
                                                self.value_at(r).reshape(X1.shape), 
                                                cmap=cm.Blues, **plot_kwargs)

            cax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
            fig.colorbar(cs, cax=cax)

            # Label the axes
            axes[0,0].set_ylabel("{0} [{1}]".format("y", x.unit))
            axes[1,0].set_xlabel("{0} [{1}]".format("x", x.unit))
            axes[1,0].set_ylabel("{0} [{1}]".format("z", x.unit))
            axes[1,1].set_xlabel("{0} [{1}]".format("y", x.unit))

        elif self.ndim == 2:
            raise NotImplementedError()
            bottom = coord_array[:, 0]
            side = coord_array[:, 1]
            X, Y = np.meshgrid(bottom,side)
            cs = axes.contourf(X, Y, self.value_at(X.ravel(),Y.ravel()).reshape(X.shape), cmap=cm.Blues, **kwargs)
            fig.colorbar(cs, shrink=0.9)
        elif self.ndim == 1:
            raise NotImplementedError()
            axes.plot(coord_array, self.value_at(coord_array))

        fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.08, bottom=0.08, top=0.9, right=0.9 )
        fig.suptitle(self._repr_latex_(), fontsize=24)

        return fig, axes

    def __add__(self, other):
        """ Allow adding two potentials """

        if not isinstance(other, CartesianPotential):
            raise TypeError("Addition is only supported between two Potential objects!")

        new_potential = CartesianPotential(self.unit_bases)
        for key in self._potential_components.keys():
            new_potential.add_component(key, self._potential_components[key], 
                                        f_prime=self._potential_component_derivs[key],
                                        parameters=self.parameters[key])

        for key in other._potential_components.keys():
            new_potential.add_component(key, other._potential_components[key], 
                                        f_prime=other._potential_component_derivs[key],
                                        parameters=other.parameters[key])

        return new_potential
