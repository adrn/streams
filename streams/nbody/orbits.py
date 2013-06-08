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

from .core import _validate_quantity
from ..misc.units import UnitSystem
from ..plot.data import scatter_plot_matrix

__all__ = ["OrbitCollection"]

class OrbitCollection(object):
        
    def __init__(self, t, r, v, m=None, unit_system=None):
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

        if not r.value.shape == v.value.shape:
            raise ValueError("Shape of position should match shape of velocity")
        
        if not len(t.value) == self.ntimesteps:
            raise ValueError("Length of time array must match number of "
                             "timesteps in position and velocity.")
        
        for x in ['r', 'v', 'm', 't']:
            setattr(self, "_{0}".format(x), 
                    eval(x).decompose(bases=unit_system).value)
        
        self.unit_system = unit_system
    
    @property
    def r(self):
        return self._r * self.unit_system['length']
    
    @property
    def v(self):
        return self._v * self.unit_system['length'] / self.unit_system['time']
    
    @property
    def m(self):
        return self._m * self.unit_system['mass']
    
    @property
    def t(self):
        return self._t * self.unit_system['time']
    