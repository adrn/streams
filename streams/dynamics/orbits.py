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

from .core import _validate_quantity, DynamicalBase
#from .particles import ParticleCollection
from ..misc.units import UnitSystem

__all__ = ["OrbitCollection"]

class OrbitCollection(DynamicalBase):
        
    def __init__(self, t, r, v, m=None, dr=None, dv=None, unit_system=None):
        """ Represents a collection of orbits, e.g. positions and velocities 
            over time for a set of particles.
            
            Input r and v should be shape (len(t), Nparticles, Ndim).
            
            Parameters
            ---------- 
            t : astropy.units.Quantity
                Time.
            r : astropy.units.Quantity
                Position.
            v : astropy.units.Quantity
                Velocity.
            dr : astropy.units.Quantity (optional)
                Uncertainty in position.
            dv : astropy.units.Quantity (optional)
                Uncertainty in velocity.
            m : astropy.units.Quantity (optional)
                Mass.
            unit_system : UnitSystem (optional)
                The desired unit system for the particles. If not provided, will
                use the units of the input Quantities.
        """
        
        _validate_quantity(t, unit_like=u.s)
        _validate_quantity(r, unit_like=u.km)
        _validate_quantity(v, unit_like=u.km/u.s)
        
        try:
            self.ntimesteps, self.nparticles, self.ndim = r.value.shape
        except ValueError:
            raise ValueError("Position and velocity should have shape "
                             "(ntimesteps, nparticles, ndim)")
        
        if unit_system is None and m is None:
            raise ValueError("If not specifying a unit_system, you must "
                             "specify a mass Quantity for the particles.")
        elif unit_system is not None and m is None:
            m = [0.] * self.nparticles * unit_system['mass']

        _validate_quantity(m, unit_like=u.kg)
        
        if unit_system is None:
            _units = [r.unit, m.unit, t.unit] + v.unit.bases
            unit_system = UnitSystem(*set(_units))

        if r.value.shape != v.value.shape:
            raise ValueError("Position and velocity must have same shape.")
        
        if len(t.value) != self.ntimesteps:
            raise ValueError("Length of time array must match number of "
                             "timesteps in position and velocity.")
        
        _r = r.decompose(unit_system).value
        _v = v.decompose(unit_system).value
        self._m = m.decompose(unit_system).value
        self._t = t.decompose(unit_system).value
        
        if dr is not None:
            _validate_quantity(dr, unit_like=u.km)
            assert dr.value.shape == r.value.shape
            _dr = dr.decompose(unit_system).value
        else:
            _dr = np.zeros_like(_r)
        
        if dv is not None:
            _validate_quantity(dv, unit_like=u.km/u.s)
            assert dv.value.shape == v.value.shape
            _dv = dv.decompose(unit_system).value
        else:
            _dv = np.zeros_like(_v)
        
        # create container for all 6 phasespace 
        self._x = np.zeros((self.ntimesteps, self.nparticles, self.ndim*2))
        self._x[..., :self.ndim] = _r
        self._x[..., self.ndim:] = _v
        
        self._dx = np.zeros((self.ntimesteps, self.nparticles, self.ndim*2))
        self._dx[..., :self.ndim] = _dr
        self._dx[..., self.ndim:] = _dv
        
        self.unit_system = unit_system
    
    @property
    def t(self):
        return self._t * self.unit_system['time']
    
    def to(self, unit_system):
        """ Return a new ParticleCollection in the specified unit system. """
        new_r = self.r.decompose(unit_system)
        new_v = self.v.decompose(unit_system)
        new_m = self.m.decompose(unit_system)
        new_t = self.t.decompose(unit_system)
        
        return OrbitCollection(t=new_t, r=new_r, v=new_v, m=new_m, 
                               unit_system=unit_system)
    
    def __getitem__(self, key):
        """ Slice on time """
        
        if isinstance(key, slice) :
            #Get the start, stop, and step from the slice
            return OrbitCollection(t=self.t[key], r=self.r[key], 
                                   v=self.v[key], m=self.m[key])
            
        elif isinstance(key, int) :
            return ParticleCollection(r=self.r[key], v=self.v[key], 
                                      m=self.m[key])
        
        else:
            raise TypeError, "Invalid argument type."
    