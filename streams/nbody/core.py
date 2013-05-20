# coding: utf-8

""" Direct N-body """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G
from astropy.utils.misc import isiterable

__all__ = ["Particle", "ParticleCollection"]

def _validate_quantity(q, unit_like=None):
    """ Validate that the input is a Quantity object. An optional parameter is
        'unit_like' which will require that the input Quantity object has a
        unit equivalent to 'unit_like'.
    """
    if not isinstance(q, u.Quantity):
        msg = "Input must be a Quantity object, not {0}.".format(type(q))
        raise TypeError(msg)
        
    elif not q.unit.is_equivalent(unit_like):
        if unit_like.physical_type != "unknown":
            msg = "Quantity must have a unit equivalent to '{0}'".format(unit_like)
        else:
            msg = "Quantity must be of type '{0}'".format(unit_like.physical_type)
        raise ValueError(msg)
        
class Particle(object):
    
    def __init__(self, r, v, m):
        """ Represents a massive particle at a given position with a velocity.
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position.
            v : astropy.units.Quantity
                Velocity.
            m : astropy.units.Quantity
                Mass.
        """
        
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
    
    def __init__(self, particles=None, r=None, v=None, m=None, units=None):
        """ TODO: """
        
        required_types = ["time", "length", "mass"]
        if units == None or not isiterable(units):
            raise ValueError("You must specify a system of units as a list of "
                             "unit bases.")
        else:
            ptypes = [x.physical_type for x in units]
            
            # multiple units of same type
            if len(set(ptypes)) != len(ptypes):
                raise ValueError("You may only specify one unit for each "
                                 "physical type.")
            
            for rt in required_types:
                if rt not in ptypes:
                    raise ValueError("You must specify a '{0}' unit!".format(rt))
        
        self._units = dict([(x.physical_type, x) for x in units])
        
        # If particles are specified, loop over the particles and separate
        #   the attributes into collection attributes
        if particles is not None:
            self.ndim = particles[0].ndim
            
            # Empty containers 
            self._r = np.zeros(shape=(len(particles), self.ndim))
            self._v = np.zeros(shape=(len(particles), self.ndim))
            self._m = np.zeros(shape=(len(particles),))
            
            for ii,particle in enumerate(particles):
                if particle.ndim != self.ndim:
                    raise ValueError("Particle {0} has {1} dimensions, others "
                                     "have {2} dimensions!".format(ii, 
                                                                   particle.ndim,
                                                                   self.ndim))
                    
                # Loop over attributes of the particles
                for k in ["r", "v", "m"]:
                    v = getattr(particle, k)
                    val = v.decompose(bases=self._units.values()).value
                    getattr(self, "_" + k)[ii] = val
        
        else:
            if r is None or v is None or m is None:
                raise ValueError("If not specfying particles, must specify "
                                 "r, v, and m (positions, velocities, and "
                                 "masses) for all particles.")
            
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
            collection of particles. 
        """
        _validate_quantity(r, unit_like=u.km)
        _validate_quantity(m, unit_like=u.kg)
        
        a = self._acceleration_at(r.decompose(bases=self._units.values()).value,
                                  m.decompose(bases=self._units.values()).value,
                                  G.decompose(bases=self._units.values()).value)
        return a * self._units['length'] / self._units['time']**2
    
    def _acceleration_at(self, _r, _m, _G):
        """ """
        if _r.ndim == 1:
            rr = _r - self._r
            a = _G * _m * self._m * rr / np.linalg.norm(rr)**3  
            return np.sum(a, axis=0)
            
        elif _r.ndim == 2:
            # could do: (np.repeat(_r[np.newaxis], self._r.shape[0], axis=0) - \
            #                self._r[...,np.newaxis]).sum(axis=0)
            a = np.zeros_like(_r)
            for ii in range(_r.shape[0]):
                rr = _r[ii] - self._r
                a[ii] = (_G * _m[ii] * self._m * rr / np.sum(rr**2,axis=0)**1.5).sum(axis=0)
        
            return a
            
        else:
            raise ValueError()
            