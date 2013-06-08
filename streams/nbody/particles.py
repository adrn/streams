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

from .core import _validate_quantity
from ..misc.units import UnitSystem
from ..plot.data import scatter_plot_matrix

__all__ = ["Particle", "ParticleCollection"]
    
class Particle(object):
    
    def __init__(self, r, v, m=None):
        """ Represents a particle at a given position with a velocity.
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position.
            v : astropy.units.Quantity
                Velocity.
            m : astropy.units.Quantity (optional)
                Mass.
        """
        
        # If mass is not specified, assume it is a test particle with no mass
        if m is None:
            m = 0.*u.kg
        
        _validate_quantity(r, unit_like=u.km)
        _validate_quantity(v, unit_like=u.km/u.s)
        _validate_quantity(m, unit_like=u.kg)
        
        self.r = r
        self.v = v
        self.m = m
        
        if self.r.value.ndim > 1 or self.v.value.ndim > 1:
            raise ValueError("The Particle class represents a single particle."
                             "For a collection of particles, use "
                             "ParticleCollection.")
        
        if self.r.value.shape != self.v.value.shape:
            raise ValueError("Position vector and velocity vector must have the"
                             "same shape ({0} vs {1})".format(self.r.value.shape,
                                                              self.v.value.shape))
        
        # dimensionality
        self.ndim = len(self.r)

class ParticleCollection(object):
    
    def __init__(self, particles=None, r=None, v=None, m=None, unit_system=None):
        """ A collection of Particles. Stores values as arrays internally
            so it is faster than dealing with a list of Particle objects.
            
            Parameters
            ---------- 
            particles : list of Particle objects
            
            or
            
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
        
        # If particles are specified, loop over the particles and separate
        #   the attributes into collection attributes
        if particles is not None:
            self.ndim = particles[0].ndim
            
            # Empty containers 
            self._r = np.zeros(shape=(len(particles), self.ndim))
            self._v = np.zeros(shape=(len(particles), self.ndim))
            self._m = np.zeros(shape=(len(particles),))
            
            for ii,particle in enumerate(particles):
                if unit_system is None:
                    unit_system = UnitSystem(particle.r.unit,
                                             particle.v.unit,
                                             particle.m.unit)
                
                if particle.ndim != self.ndim:
                    raise ValueError("Particle {0} has {1} dimensions, others "
                                     "have {2} dimensions!".format(ii, 
                                                                   particle.ndim,
                                                                   self.ndim))
                    
                # Loop over attributes of the particles
                for k in ["r", "v", "m"]:
                    v = getattr(particle, k)
                    val = v.decompose(unit_system).value
                    getattr(self, "_" + k)[ii] = val
        
        else:
            if r is None or v is None:
                raise ValueError("If not specfying particles, must specify "
                                 "r, and v (positions, velocities) for all "
                                 "particles.")
            
            if m is None:
                m = [0.]*len(r)*unit_system['mass']
            
            _validate_quantity(r, unit_like=u.km)
            _validate_quantity(v, unit_like=u.km/u.s)
            _validate_quantity(m, unit_like=u.kg)
            
            if r.value.ndim < 2:
                raise ValueError("ParticleCollection must contain more than one"
                                 " particle!")
            
            self.ndim = r.value.shape[1]

            assert r.value.shape == v.value.shape
            
            for x in ['r', 'v', 'm']:
                setattr(self, "_{0}".format(x), 
                        eval(x).decompose(bases=self._units.values()).value)
        
        # TODO:
        #if unit_system is not None
        
        # Create internal G in the correct unit system
        self._G = G.decompose(bases=self._units.values()).value
    
    @property
    def units(self):
        return self._units.values()
    
    @property
    def r(self):
        return self._r * self._units['length']
    
    @property
    def v(self):
        return self._v * self._units['length'] / self._units['time']
    
    @property
    def m(self):
        return self._m * self._units['mass']
        
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
        
        a = self._acceleration_at(r.decompose(bases=self._units.values()).value,
                                  m.decompose(bases=self._units.values()).value,
                                  G.decompose(bases=self._units.values()).value)
        return a * self._units['length'] / self._units['time']**2
    
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
    
    def __repr__(self):
        return "<ParticleCollection N={0}>".format(len(self._r))
    
    def plot_r(self, coord_names, **kwargs):
        """ Make a scatter-plot of 3 projections of the positions of the 
            particle coordinates.
            
            Parameters
            ----------
            coord_names : list
                Name of each axis, e.g. ['x','y','z']
            kwargs (optional)
                Keyword arguments that get passed to scatter_plot_matrix
        """   
        assert len(coord_names) == self.ndim, "Must pass a coordinate name for each dimension."
        
        labels = [r"{0} [{1}]".format(nm, self.r.unit)
                    for nm in coord_names]
        
        fig,axes = scatter_plot_matrix(self._r.T, 
                                       labels=labels,
                                       **kwargs)
        return fig, axes
    
    def plot_v(self, coord_names, **kwargs):
        """ Make a scatter-plot of 3 projections of the velocities of the 
            particle coordinates.
            
            Parameters
            ----------
            coord_names : list
                Name of each axis, e.g. ['Vx','Vy','Vz']
            kwargs (optional)
                Keyword arguments that get passed to scatter_plot_matrix
        """   
        assert len(coord_names) == self.ndim, "Must pass a coordinate name for each dimension."
        
        labels = [r"{0} [{1}]".format(nm, self.v.unit)
                    for nm in coord_names]
        
        fig,axes = scatter_plot_matrix(self._v.T, 
                                       labels=labels,
                                       **kwargs)
        return fig, axes
    
    def merge(self, other):
        """ Merge two particle collections"""
        
        if not isinstance(other, ParticleCollection):
            raise TypeError("Can only merge two ParticleCollection objects!")
        
        if other._units != self._units:
            raise ValueError("Unit systems much match!")
        
        r = np.vstack((self._r,other._r)) * self._units['length']
        v = np.vstack((self._v,other._v)) * self._units['length']/self._units['time']
        m = np.append(self._m,other._m) * self._units['mass']
        
        return ParticleCollection(r=r, v=v, m=m, units=self.units)
