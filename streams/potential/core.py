# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
sys.path.append("/Users/adrian/projects/astropy_adrn")
import copy
import inspect

# Third-party
import numpy as np
from astropy.utils.misc import isiterable

def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])

class Potential(object):

    def __init__(self):
        self._potential_components = dict()
        self._potential_component_derivs = dict()
        self._latex = dict()

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
                accelerations[:,ii] += - potential_derivative(*coord_array.T)
        print(accelerations.shape)
        return accelerations

    def _repr_latex_(self):
        ''' Generate a latex representation of the potential. This is used by the
            IPython notebook to render nice latex equations.
        '''

        ltx_str = ""
        for name,latex in self._latex.items():
            ltx_str += "\\textit{{{0}}}: {1}\\\\".format(name, latex)

        return u'Components: ${0}$'.format(ltx_str)

    def _args_to_coords(self, args):
        ''' Private method to convert a list of arguments into coordinates that Potential understands '''

        if len(args) == 1 and isinstance(args[0], np.ndarray):
            coord_array = args[0]

            if len(coord_array.shape) == 1:
                coord_array = coord_array[:,np.newaxis]

            if not coord_array.shape[1] == self.ndim:
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