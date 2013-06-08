# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

__all__ = ["UnitSystem"]

required_units = ['length', 'mass', 'time']
class UnitSystem(dict):
    
    def __init__(self, *bases):
        """ Given Unit objects as positional arguments, defines a 
            system of physical units. At minimum, must contain length,
            time, and mass.
        """
        
        # Internal registry
        _reg = dict()
        
        # For each unit provided, store it in the registry keyed by the
        #   physical type of the unit
        for ubase in bases:
            try:
                ptype = ubase.physical_type
            except AttributeError:
                raise TypeError("Non-standard Unit object '{0}'".format(ubase))
            
            if _reg.has_key(ptype):
                raise ValueError("Multiple units provided for physical type: "
                                 "'{0}'".format(ptype))
            
            _reg[ptype] = ubase
        
        # Check to make sure each of the required physical types is provided
        for runit in required_units:
            if runit not in _reg.keys():
                raise ValueError("Must define, at minimum, a system with "
                                 "{0}".format(','.join(required_units)))
        
        super(UnitSystem, self).__init__(**_reg)
        
    @property
    def bases(self):
        return self.values()
    
    def __eq__(self, other):
        for rtype in required_units:
            if self[rtype] != other[rtype]:
                return False
        
        return True
    
    def __iter__(self):
        return iter(self.bases)