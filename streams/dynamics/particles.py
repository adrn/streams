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

__all__ = ["Particle"]
    
class Particle(DynamicalBase):
    
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
            
        if r.value.shape != v.value.shape:
            raise ValueError("Position and velocity must have same shape.")
        
        # decompose each input into the specified unit system
        _r = r.decompose(self.units).value
        _v = v.decompose(self.units).value
        
        # create container for all 6 phasespace 
        self._x = np.zeros((self.nparticles, self.ndim*2))
        self._x[:,:self.ndim] = _r
        self._x[:,self.ndim:] = _v
        self._m = m.decompose(self.units).value
        
        # Create internal G in the correct unit system for speedy acceleration
        #   computation
        self._G = G.decompose(self.units).value
    
    def to(self, units):
        """ Return a new Particle in the specified unit system. """
        
        return Particle(r=self.r, v=self.v, m=self.m, units=units)

    def acceleration_at(self, r):
        """ Compute the acceleration at a given position due to the 
            collection of particles. Inputs must be Quantity objects.
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position.
                
        """
        _validate_quantity(r, unit_like=u.km)
        
        r_unit = filter(lambda x: x.is_equivalent(u.km), self.units)[0]
        t_unit = filter(lambda x: x.is_equivalent(u.s), self.units)[0]

        a = self._acceleration_at(r.decompose(self.units).value)
        return a * r_unit / t_unit**2
    
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
                a[ii] = (self._G * self._m * rr / np.sum(rr**2,axis=0)**1.5)\
                        .sum(axis=0)
        
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