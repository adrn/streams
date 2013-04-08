# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable

from ..potential import Potential
from ..integrate import leapfrog

__all__ = ["Particle", "Orbit"]

class Particle(object):
    
    def __init__(self, r, v, m):
        """ Represents a single or array of massive particles with positions 
            and velocities. Useful for integration because internally stores 
            data as arrays.
            
            Parameters
            ----------
            r : astropy.units.Quantity
                A Quantity object with position(s) aligned with the
                specified velocities.
            v : astropy.units.Quantity
                A Quantity object with velocities aligned with the
                specified position(s).
            m : astropy.units.Quantity
                A Quantity object with an array of masses aligned with the
                specified positions/velocities.
        """
                
        if not isinstance(r, u.Quantity) or \
           not isinstance(v, u.Quantity) or \
           not isinstance(m, u.Quantity):
            raise TypeError("Position, Velocity, and Mass must be Astropy "
                            "Quantity objects. You specified {0},{1},{2}."
                            .format(type(r),type(v),type(m)))
        
        if not isiterable(r.value):
            r = u.Quantity([r.value], unit=r.unit)
        
        if not isiterable(v.value):
            v = u.Quantity([v.value], unit=v.unit)
        
        if not isiterable(m.value):
            m = u.Quantity([m.value], unit=m.unit)
        
        if r.value.shape != v.value.shape:
            raise ValueError("Shape mismatch: position and velocity "
                             "shapes must match. {0}, {1}"
                             .format(r.value.shape,v.value.shape))
        
        if len(r.value.shape) > 1:
            if r.value.shape[0] != m.value.shape[0]:
                raise ValueError("Shape mismatch: mass array must be aligned "
                                 "with positions along axis=0. {0} vs {1}"
                                 .format(m.value.shape, r.value.shape))
                             
        self.r = r
        self.v = v
        self.m = m
    
    def __getitem__(self, val):
        """ Implement indexing / slicing """
        
        if isinstance(val, int) or isinstance(val, slice):
            return Particle(r=self.r[val], v=self.v[val], m=self.m[val])
        else:
            raise TypeError("Indices must be integer, not {0}".format(type(val)))
    
    def __len__(self):
        return len(self.r)
    
    def integrate(self, potential, t, integrator=leapfrog, nbody=False):
        """ Integrate the particle(s) from time t1 to t2 in the given 
            potential. 
            
            Parameters
            ----------
            potential : streams.Potential
                A Potential object to integrate the particles under.
            t : numpy.ndarray
                An array of times to integrate the particles on.
            integrator : func (optional)
                A function to use for integrating the particle orbits.
            nbody : bool (optional)
                Use N-body integration, computing inter-particle forces. 
        """
        
        if not isinstance(potential, Potential):
            raise TypeError()
        
        if nbody == False:
            # test particles -- ignore particle-particle effects
            t, r, v = integrator(self.potential.acceleration_at, 
                                 self.r,
                                 self.v, 
                                 t=t)
        else:
            raise NotImplementedError("nbody integration not yet supported")
        
class Orbit(object):
    
    def __init__(self, t):
        """ Represents the orbit of a Particle """