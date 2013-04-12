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

__all__ = ["CartesianPotential", "CompositePotential"]

class Potential(object):
    
    def _validate_unit_system(self, unit_bases):
        """ Given a list or dictionary of astropy unit objects, make sure it is
            a valid system of units: e.g., make sure it contains a length, mass,
            and time (at minimum).
        """
        
        try:
            unit_bases = unit_bases.values()
        except AttributeError:
            pass
        
        # Check that unit_bases contains at least a length, mass, and time unit
        _base_map = dict()
        for base in unit_bases:
            tp = base.physical_type
            if tp in _base_map.keys():
                raise ValueError("You specified multiple {0} units! Only one "
                                 "unit per physical type permitted."
                                 .format(tp))
            
            # hack because v0.2 of Astropy doesn't have a 'time' physical type
            if tp == "unknown":
                if base.is_equivalent(u.s):
                    tp = "time"
                else:
                    raise ValueError("Unknown physical type for unit {0}."
                                     .format(base))
            
            _base_map[tp] = base
        
        if "length" not in _base_map.keys() or \
           "mass" not in _base_map.keys() or \
           "time" not in _base_map.keys():
            raise ValueError("You must specify, at minimum, a length unit, "
                             "mass unit, and time unit.")
        
        return _base_map
    
    def _validate_origin(self, origin):
        """ Make sure the origin is a Quantity and has length-like units """
        
        try:
            if origin.unit.physical_type != "length":
                raise ValueError("origin must have length units, not {0}."
                                 .format(origin.unit.physical_type))
        except AttributeError:
            raise TypeError("origin must be an Astropy Quantity-like object. "
                            "You passed a {0}.".format(type(origin)))
        
        return origin

class CartesianPotential(Potential):

    def __init__(self, units, f, f_prime, latex=None, parameters=None, 
                 origin=None):
        """ A baseclass for representing gravitational potentials in Cartesian
            coordinates. You must specify the functional form of the potential
            component. You may also optionally add derivatives using the 
            f_prime keyword.
            
            Note::
                Currently only supports 3D potentials.

            Parameters
            ----------
            units : list, dict
                Either a list or dictionary of base units specifying the 
                system of units for this potential. 
            f : function
                The functional form of the potential component. This must be a
                function that accepts N arguments where N is the dimensionality.
            f_prime : tuple (optional)
                A functions that computes the derivatives of the potential.
            latex : str (optional)
                The latex representation of this potential component. Will be
                used to make sexy output in iPython Notebook.
            parameters : dict (optional)
                Any extra parameters that the potential function requires.
            origin : astropy.units.Quantity (optional)
                Must specify the location of the potential origin along each 
                dimension. For example, it could look like 
                    origin=[0,0,0]*u.kpc
            
        """
                
        self.units = self._validate_unit_system(units)
        
        if origin == None:
            origin = [0.,0.,0.] * self.units["length"]
        
        self.origin = self._validate_origin(origin)
        self.ndim = len(self.origin)       
        
        self._unscaled_parameters = parameters
        self.parameters = self._rescale_parameters(parameters)
        self._scaled_origin = self.origin.decompose(bases=self.units.values()).value
        
        # Make sure the f is callable, and that the component doesn't already
        #   exist in the potential
        if not hasattr(f, '__call__'):
            raise TypeError("'f' parameter must be a callable function! You "
                            "passed in a '{0}'".format(f.__class__))
        
        self.f = lambda *args: f(*args, 
                                 origin=self._scaled_origin, 
                                 **self.parameters)
        
        if f_prime != None:
            if not hasattr(f_prime, '__call__'):
                raise TypeError("'f_prime' must be a callable function! You "
                                "passed in a '{0}'".format(f_prime.__class__))
            self.f_prime = lambda *args: f_prime(*args, 
                                 origin=self._scaled_origin, 
                                 **self.parameters)
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
                _params[param_name] = val.decompose(bases=self.units.values()).value
            except AttributeError: # not Quantity-like
                _params[param_name] = val
        
        return _params
    
    def value_at(self, r):
        """ Compute the value of the potential at the given position(s) 
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the value at.
        """
        
        x,y,z = self._r_to_xyz(r)
        return self.f(x,y,z)

    def acceleration_at(self, r):
        """ Compute the acceleration due to the potential at the given 
            position(s) 
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the acceleration at.
        """
        
        x,y,z = self._r_to_xyz(r)
        return -self.f_prime(x,y,z)

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

    