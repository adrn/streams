# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from abc import ABCMeta, abstractproperty, abstractmethod

# Third-party
import astropy.units as u

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

class DynamicalBase(object):
    __metaclass__ = ABCMeta
    
    @property
    def _r(self):
        return self._x[...,:self.ndim]
    
    @property
    def _v(self):
        return self._x[...,self.ndim:]
    
    @property
    def r(self):
        r_unit = filter(lambda x: x.is_equivalent(u.km), self.units)[0]
        return self._r * r_unit
    
    @property
    def v(self):
        r_unit = filter(lambda x: x.is_equivalent(u.km), self.units)[0]
        t_unit = filter(lambda x: x.is_equivalent(u.s), self.units)[0]
        return self._v * r_unit / t_unit
    
    @property
    def m(self):
        m_unit = filter(lambda x: x.is_equivalent(u.kg), self.units)[0]
        return self._m * m_unit
    
    @abstractmethod
    def to(self, units):
        pass
        
    def __repr__(self):
        return "<{0} N={1}>".format(self.__class__.__name__, 
                                    self.nparticles)