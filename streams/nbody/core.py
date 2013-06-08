# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G
from astropy.utils.misc import isiterable

__all__ = ["Orbit", "OrbitCollection"]

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

class Orbit(object):
    
    def __init__(self, t, r, v, m):
        """ Represents a massive particle at a given position with a velocity.
            
            Parameters
            ----------
            t : astropy.units.Quantity
                Time.
            r : astropy.units.Quantity
                Position.
            v : astropy.units.Quantity
                Velocity.
            m : astropy.units.Quantity
                Mass.
        """
        
        _validate_quantity(t, unit_like=u.s)
        _validate_quantity(r, unit_like=u.km)
        _validate_quantity(v, unit_like=u.km/u.s)
        _validate_quantity(m, unit_like=u.kg)
        
        self.t = t
        self.r = r
        self.v = v
        self.m = m
        
        if self.r.value.ndim > 2 or self.v.value.ndim > 2:
            raise ValueError("The Orbit class represents a single orbit."
                             "For a collection of orbits, use "
                             "OrbitCollection.")
        
        if self.r.value.shape != self.v.value.shape:
            raise ValueError("Position vector and velocity vector must have the"
                             "same shape ({0} vs {1})".format(self.r.value.shape,
                                                              self.v.value.shape))
        
        # dimensionality
        self.ndim = self.r.value.shape[1]

class OrbitCollection(object):
        
    def __init__(self, orbits=None, t=None, r=None, v=None, m=None, units=None):
        """ Represents a collection of orbits, e.g. positions and velocities 
            over time for a set of particles.
            
            Input r and v should be shape (len(t), Nparticles, Ndim)
        """
        
        self._units = _validate_units(units)
        
        if orbits is not None:
            # TODO
            raise NotImplementedError()
        else:
            if r is None or v is None or t is None:
                raise ValueError("If not specfying orbits, must specify "
                                 "r, v, and t (positions, velocities, and time).")
            
            if m is None:
                m = [0.]*r.shape[1]
            
            _validate_quantity(r, unit_like=u.km)
            _validate_quantity(v, unit_like=u.km/u.s)
            _validate_quantity(m, unit_like=u.kg)
            _validate_quantity(t, unit_like=u.s)
            
            if r.value.ndim < 3:
                raise ValueError("OrbitCollection must contain more than one"
                                 " orbit!")
            
            self.ndim = r.value.shape[2]

            assert r.value.shape == v.value.shape
            assert t.value.shape[0] == r.value.shape[0]
            
            for x in ['r', 'v', 'm', 't']:
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
    
    @property
    def t(self):
        return self._t * self._units['time']
    