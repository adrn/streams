# coding: utf-8

""" Massive particles or collections of particles """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

from .core import _validate_quantity, DynamicalBase
from ..misc.units import UnitSystem

__all__ = ["ParticleCollection"]
    
class ParticleCollection(DynamicalBase):
    
    def __init__(self, r, v, m=None, unit_system=None):
        """ A collection of massive or test particles. 
            
            Parameters
            ---------- 
            r : astropy.units.Quantity
                Position.
            v : astropy.units.Quantity
                Velocity.
            m : astropy.units.Quantity (optional)
                Mass.
            unit_system : UnitSystem (optional)
                The desired unit system for the particles. If not provided, will
                use the units of the input Quantities.
        """
        
        _validate_quantity(r, unit_like=u.km)
        _validate_quantity(v, unit_like=u.km/u.s)
        
        try:
            self.nparticles, self.ndim = r.value.shape
        except ValueError:
            raise ValueError("Position and velocity should have shape "
                             "(nparticles, ndim)")
        
        if unit_system is None and m is None:
            raise ValueError("If not specifying a unit_system, you must "
                             "specify a mass Quantity for the particles.")
        elif unit_system is not None and m is None:
            m = [0.]*len(r.value)*unit_system['mass']
            
        _validate_quantity(m, unit_like=u.kg)
        
        if unit_system is None:
            _units = [r.unit, m.unit] + v.unit.bases
            unit_system = UnitSystem(*set(_units))
            
        if r.value.shape != v.value.shape:
            raise ValueError("Position and velocity must have same shape.")
        
        # decompose each input into the specified unit system
        _r = r.decompose(unit_system).value
        _v = v.decompose(unit_system).value
        self._m = m.decompose(unit_system).value
        
        # create container for all 6 phasespace 
        self._x = np.zeros((self.nparticles, self.ndim*2))
        self._x[:,:self.ndim] = _r
        self._x[:,self.ndim:] = _v
        
        # Create internal G in the correct unit system for speedy acceleration
        #   computation
        self._G = G.decompose(unit_system).value
        self.unit_system = unit_system
    
    def to(self, unit_system):
        """ Return a new ParticleCollection in the specified unit system. """
        new_r = self.r.decompose(unit_system)
        new_v = self.v.decompose(unit_system)
        new_m = self.m.decompose(unit_system)
        
        return ParticleCollection(r=new_r, v=new_v, m=new_m, 
                                  unit_system=unit_system)
    
    def acceleration_at(self, r, m):
        """ Compute the acceleration at a given position due to the 
            collection of particles. Inputs must be Quantity objects.
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position.
            v : astropy.units.Quantity
                Velocity.
                
        """
        _validate_quantity(r, unit_like=u.km)
        _validate_quantity(m, unit_like=u.kg)
        
        a = self._acceleration_at(r.decompose(self.unit_system).value)
        return a * self.unit_system['length'] / self.unit_system['time']**2
    
    def _acceleration_at(self, _r):
        """ Compute the acceleration at a given position due to the 
            collection of particles. Inputs are arrays, and assumes they 
            are in the correct system of units.
        """
        if _r.ndim == 1:
            rr = _r - self._r
            a = self._G * self._m * rr / np.linalg.norm(rr)**3  
            return np.sum(a, axis=0)
            
        elif _r.ndim == 2:
            # could do: (np.repeat(_r[np.newaxis], self._r.shape[0], axis=0) - \
            #                self._r[...,np.newaxis]).sum(axis=0)
            #   but this involves making a potentially big array in memory...
            a = np.zeros_like(_r)
            for ii in range(_r.shape[0]):
                rr = _r[ii] - self._r
                a[ii] = (self._G * self._m * rr / np.sum(rr**2,axis=0)**1.5).sum(axis=0)
        
            return a
            
        else:
            raise ValueError()
    
    def plot_r(self, coord_names=['x','y','z'], **kwargs):
        """ Make a scatter-plot of 3 projections of the positions of the 
            particle coordinates.
            
            Parameters
            ----------
            coord_names : list
                Name of each axis, e.g. ['x','y','z']
            kwargs (optional)
                Keyword arguments that get passed to scatter_plot_matrix
        """   
        from ..plot.data import scatter_plot_matrix
        if not len(coord_names) == self.ndim:
            raise ValueError("Must pass a coordinate name for each dimension.")
        
        labels = [r"{0} [{1}]".format(nm, self.r.unit)
                    for nm in coord_names]
        
        fig,axes = scatter_plot_matrix(self._r.T, 
                                       labels=labels,
                                       **kwargs)
        return fig, axes
    
    def plot_v(self, coord_names=['vx','vy','vz'], **kwargs):
        """ Make a scatter-plot of 3 projections of the velocities of the 
            particle coordinates.
            
            Parameters
            ----------
            coord_names : list
                Name of each axis, e.g. ['Vx','Vy','Vz']
            kwargs (optional)
                Keyword arguments that get passed to scatter_plot_matrix
        """   
        from ..plot.data import scatter_plot_matrix
        assert len(coord_names) == self.ndim, "Must pass a coordinate name for each dimension."
        
        labels = [r"{0} [{1}]".format(nm, self.v.unit)
                    for nm in coord_names]
        
        fig,axes = scatter_plot_matrix(self._v.T, 
                                       labels=labels,
                                       **kwargs)
        return fig, axes
    
    def merge(self, other):
        """ Merge two particle collections. Takes unit system from the first
            ParticleCollection.
        """
        
        if not isinstance(other, ParticleCollection):
            raise TypeError("Can only merge two ParticleCollection objects!")
        
        other_r = other.r.decompose(self.unit_system).value
        other_v = other.v.decompose(self.unit_system).value
        other_m = other.m.decompose(self.unit_system).value
        
        r = np.vstack((self._r,other_r)) * self.unit_system['length']
        v = np.vstack((self._v,other_v)) * self.unit_system['length']/self.unit_system['time']
        m = np.append(self._m,other_m) * self.unit_system['mass']
        
        return ParticleCollection(r=r, v=v, m=m, unit_system=self.unit_system)
    
    def __getitem__(self, key):
    
        r = self._r[key] * self.unit_system['length']
        v = self._v[key] * self.unit_system['length']/self.unit_system['time']
        m = self._m[key] * self.unit_system['mass']
        
        return ParticleCollection(r=r, v=v, m=m, unit_system=self.unit_system)
