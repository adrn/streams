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

__all__ = ["UnitSystem", "CartesianPotential", "CompositePotential"]

required_units = ['length', 'mass', 'time']
class UnitSystem(object):
    
    def __init__(self, *bases):
        """ Given Unit objects as positional arguments, defines a 
            system of physical units. At minimum, must contain length,
            time, and mass.
        """
        
        # Internal registry
        self._reg = dict()
        
        # For each unit provided, store it in the registry keyed by the
        #   physical type of the unit
        for ubase in bases:
            try:
                ptype = ubase.physical_type
            except AttributeError:
                raise TypeError("Non-standard Unit object '{0}'".format(ubase))
            
            if self._reg.has_key(ptype):
                raise ValueError("Multiple units provided for physical type: "
                                 "'{0}'".format(ptype))
            
            self._reg[ptype] = ubase
        
        # Check to make sure each of the required physical types is provided
        for runit in required_units:
            if runit not in self._reg.keys():
                raise ValueError("Must define, at minimum, a system with "
                                 "{0}".format(','.join(required_units)))
    
    @property
    def bases(self):
        return self._reg.values()

class Potential(object):

    def _validate_unit_system(self, unit_system):
        """ Make sure the provided unit_system is a UnitSystem object. """
        
        if not hasattr(unit_system, "bases"):
            raise TypeError("unit_system must be a value UnitSystem object, "
                            "or an astropy.units system.")
        
        return unit_system

class CartesianPotential(Potential):

    def __init__(self, unit_system, f, f_prime, latex=None, parameters=None):
        """ A baseclass for representing gravitational potentials in Cartesian
            coordinates. You must specify the functional form of the potential
            component. You may also optionally add derivatives using the 
            f_prime keyword.

            Parameters
            ----------
            unit_system : UnitSystem
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
            
        self.unit_system = self._validate_unit_system(unit_system)
        
        # Convert parameters to the given unit_system
        self.parameters = parameters
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
                _params[param_name] = val.decompose(bases=self.unit_system.bases).value
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
        _r = r.decompose(bases=self.unit_system.bases).value
        c = (u.J/u.kg).decompose(bases=self.unit_system.bases)
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
        
        _r = r.decompose(bases=self.unit_system.bases).value
        c = (u.m/u.s**2).decompose(bases=self.unit_system.bases)
        return self._acceleration_at(_r) * u.CompositeUnit(1., c.bases, 
                                                           c.powers)
    
    ####
    def _repr_latex_(self):
        """ Generate a latex representation of the potential. This is used by
            the IPython notebook to render nice latex equations.
        """
        # TODO: some way to also show parameter values?
        return u'${0}$'.format(self._latex)

    def _r_to_xyz(self, r):
        #if isinstance(r, u.Quantity):
        #    r = r.decompose(bases=self.units.values()).value
        
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
                                                cmap=cm.bone_r, **plot_kwargs)

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

class CompositePotential(dict, CartesianPotential):
    
    def __init__(self, units, origin, *args, **kwargs):
        """ Represents a potential composed of several sub-potentials. For 
            example, two point masses or a galactic disk + halo. The origins 
            of the components are *relative to the origin of the composite*.
            
            Parameters
            ----------
            units : list, dict
                Either a list or dictionary of base units specifying the 
                system of units for this potential. 
        """
        self.units = self._validate_unit_system(units)
        self.origin = self._validate_origin(origin)
        
        for v in kwargs.values():
            if not isinstance(v, Potential):
                raise TypeError("Values may only be Potential objects, not "
                                "{0}.".format(type(v)))
        
        self.ndim = len(self.origin)
        
        dict.__init__(self, *args, **kwargs)
    
    def __repr__(self):
        """ TODO: figure out what to display... """
        
        return "<CompositePotential ??????>"
    
    def __setitem__(self, key, value):
        if not isinstance(value, Potential):
            raise TypeError("Values may only be Potential objects, not "
                            "{0}.".format(type(value)))
        super(CompositePotential, self).__setitem__(key, value)
    
    @property
    def _latex(self):
        return "; ".join([x._latex for x in self.values()])
    
    def value_at(self, r):
        """ Compute the value of the potential at the given position(s) 
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the value at.
        """
        
        x,y,z = self._r_to_xyz(r)
        
        for potential in self.values():
            try:
                value += potential.f(x,y,z)
            except NameError:
                value = potential.f(x,y,z)
        return value

    def acceleration_at(self, r):
        """ Compute the acceleration due to the potential at the given 
            position(s) 
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the acceleration at.
        """
        
        x,y,z = self._r_to_xyz(r)
        
        for potential in self.values():
            try:
                value += -potential.f_prime(x,y,z)
            except NameError:
                value = -potential.f_prime(x,y,z)
        return value

    