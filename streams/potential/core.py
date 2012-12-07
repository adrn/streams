# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

__all__ = ["Potential"]

# Standard library
import os
import sys
sys.path.append("/Users/adrian/projects/astropy_adrn")
import copy
import inspect

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.utils.misc import isiterable
import astropy.units as u

from ..util import *
from ..coordinates import CartesianCoordinates, SphericalCoordinates, CylindricalCoordinates

class Potential(object):

    def __init__(self, length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass):
        self.length_unit = u.Unit(length_unit)
        self.time_unit = u.Unit(time_unit)
        self.mass_unit = u.Unit(mass_unit)

        self._potential_components = dict()
        self._potential_component_derivs = dict()
        self._latex = dict()
        self.coordinate_system = None

    def add_component(self, name, func, derivs=None, latex=None):
        ''' Add a component to the potential. The component must have a name, and
            you must specify the functional form of the potential component. You may
            also optionally add derivatives using the 'derivs' parameter.

            Parameters
            ----------
            name : str, hashable
                The name of the potential component, e.g. 'halo'
            func : function
                The functional form of the potential component. This must be a function
                that accepts N arguments where N is the dimensionality
            derivs : tuple
                A tuple of functions representing the derivatives of the potential.
            latex : str (optional)
                The latex representation of this potential component. Will be used to
                make sexy output in iPython Notebook.

        '''

        # Make sure the func is callable, and that the component doesn't already exist in the potential
        if not hasattr(func, '__call__'):
            raise TypeError("'func' parameter must be a callable function! You passed in a '{0}'".format(func.__class__))

        if self._potential_components.has_key(name):
            raise NameError("Potential component '{0}' already exists!".format(name))

        self._potential_components[name] = func
        self.ndim = len(inspect.getargspec(func).args)

        # If the user passes the potential derivatives, make sure it is an iterable of functions
        if derivs is not None:
            if not isiterable(derivs):
                raise TypeError("'derivs' should be a tuple of functions for the potential derivatives.")

            derivs = tuple(derivs)

            # Number of derivative functions should be equal to the dimensionality of the potential
            if len(derivs) != self.ndim:
                raise ValueError("Number of derivative functions should equal dimensionality of potential! (e.g. the number of arguments for the potential function).")

            for deriv in derivs:
                if not hasattr(deriv, '__call__'):
                    raise TypeError("'derivs' parameter must be a tuple of functions! You passed in a '{0}'".format(deriv.__class__))

        self._potential_component_derivs[name] = derivs

        if latex != None:
            if latex.startswith("$"):
                latex = latex[1:]

            if latex.endswith("$"):
                latex = latex[:-1]

            self._latex[name] = latex

    def value_at(self, *args):
        ''' Compute the value of the potential at the given position(s) '''

        coord_array = self._args_to_coords(args)

        # Compute contribution to the potential from each component
        for potential_component in self._potential_components.values():
            try:
                potential_value += potential_component(*coord_array.T)
            except NameError:
                potential_value = potential_component(*coord_array.T)

        return potential_value

    def acceleration_at(self, *args):
        ''' Compute the acceleration due to the potential at the given position(s) '''

        coord_array = self._args_to_coords(args)

        # Define empty container for accelerations
        accelerations = np.zeros_like(coord_array, dtype=float)

        for component_funcs in self._potential_component_derivs.values():
            for ii,potential_derivative in enumerate(component_funcs):
                accelerations[:,ii] -= potential_derivative(*coord_array.T)

        return accelerations

    def energy_at(self, position, velocity):
        """ Compute the total energy for an array of particles. """

        # Coordinate systems imported from ..utils
        if self.coordinate_system == CartesianCoordinates:
            kinetic = 0.5*np.sum(velocity**2, axis=2)

        elif self.coordinate_system == SphericalCoordinates:
            kinetic = 0.5*(self.vel[:,:,0]**2 + self.pos[:,:,0]**2*sin(self.vel[:,:,2])**2*self.vel[:,:,1] + self.pos[:,:,0]**2*self.vel[:,:,2]**2)

        elif self.coordinate_system == CylindricalCoordinates:
            kinetic = 0.5*(self.vel[:,:,0]**2 + self.pos[:,:,0]**2 * self.vel[:,:,1]**2 + self.vel[:,:,2]**2)

        else:
            raise ValueError("Unknown potential coordinate system '{0}'".format(self.coordinate_system))

        potential_energy = self.value_at(position)
        return kinetic + potential_energy.T

    def _repr_latex_(self):
        ''' Generate a latex representation of the potential. This is used by the
            IPython notebook to render nice latex equations.
        '''

        ltx_str = ""
        for name,latex in self._latex.items():
            ltx_str += "\\textit{{{0}}}: {1}\\\\".format(name, latex)

        if ltx_str == "":
            return ""

        return u'Components: ${0}$'.format(ltx_str)

    def _args_to_coords(self, args):
        ''' Private method to convert a list of arguments into coordinates that Potential understands '''

        if len(args) == 1 and isinstance(args[0], np.ndarray):
            coord_array = args[0]

            if len(coord_array.shape) == 1:
                coord_array = coord_array[:,np.newaxis]

            if not coord_array.shape[-1] == self.ndim:
                raise ValueError("Array of shape '{0}' does not match potential of {1} dimensions along axis 1.".format(coord_array.shape, self.ndim))

        elif len(args) == self.ndim:
            coords = [_validate_coord(x) for x in args]
            coord_len = len(coords[0])
            for coord in coords:
                if len(coord) != coord_len:
                    raise ValueError("Individual coordinate arrays have different lengths.")

            coord_array = np.vstack(coords).T

        else:
            raise ValueError("You must supply either a single position as a list of arguments, or an array where axis 0 has length 'self.ndim'.")

        return coord_array

    def plot(self, *args, **kwargs):
        ''' Plot equipotentials lines. Must pass in coordinate arrays to evaluate the
            potential over (positional args). Any keyword arguments are passed to the
            matplotlib.pyplot.contourf() function call

        '''

        coord_array = self._args_to_coords(args)

        if not kwargs.has_key("axes"):
            if self.ndim > 1:
                fig, axes = plt.subplots(self.ndim-1, self.ndim-1, sharex=True, sharey=True, figsize=(12,12))
            else:
                fig, axes = plt.subplots(1, 1, figsize=(12,12))
        else:
            axes = kwargs["axes"]
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

                    print("axes[{0},{1}] <= x:{2}, y:{3}".format(ii-1,jj,jj,ii))
                    bottom = coord_array[:, jj]
                    side = coord_array[:, ii]
                    X1, X2 = np.meshgrid(bottom,side)

                    args = [np.zeros_like(X1.ravel()) for xx in range(self.ndim)]
                    args[jj] = X1.ravel()
                    args[ii] = X2.ravel()

                    cs = axes[ii-1,jj].contourf(X1, X2, self.value_at(*args).reshape(X1.shape), cmap=cm.Blues, **kwargs)

            cax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
            fig.colorbar(cs, cax=cax)

            # Label the axes
            if self.coordinate_system != None:
                axis_names = self.coordinate_system._axis_names
                axes[0,0].set_ylabel("{1} [{0}]".format(self.length_unit, axis_names[1]))
                axes[1,0].set_xlabel("{1} [{0}]".format(self.length_unit, axis_names[0]))
                axes[1,0].set_ylabel("{1} [{0}]".format(self.length_unit, axis_names[2]))
                axes[1,1].set_xlabel("{1} [{0}]".format(self.length_unit, axis_names[1]))

        elif self.ndim == 2:
            bottom = coord_array[:, 0]
            side = coord_array[:, 1]
            X, Y = np.meshgrid(bottom,side)
            cs = axes.contourf(X, Y, self.value_at(X.ravel(),Y.ravel()).reshape(X.shape), cmap=cm.Blues, **kwargs)
            fig.colorbar(cs, shrink=0.9)
        elif self.ndim == 1:
            axes.plot(coord_array, self.value_at(coord_array))

        fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.08, bottom=0.08, top=0.9, right=0.9 )
        fig.suptitle(self._repr_latex_(), fontsize=24)

        return fig, axes

    def __add__(self, other):
        """ Allow adding two potentials """

        if not isinstance(other, Potential):
            raise TypeError("Addition is only supported between two Potential objects!")

        if other.coordinate_system != self.coordinate_system:
            raise ValueError("Potentials must have same coordinate system.")

        new_potential = Potential()
        for key in self._potential_components.keys():
            new_potential.add_component(key, self._potential_components[key], derivs=self._potential_component_derivs[key])

        for key in other._potential_components.keys():
            new_potential.add_component(key, other._potential_components[key], derivs=other._potential_component_derivs[key])

        new_potential.coordinate_system = self.coordinate_system
        return new_potential
