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

from ..potential import CartesianPotential
from ..integrate import leapfrog

__all__ = ["TestParticle", "TestParticleOrbit"]

def _validate_r_v(r,v):
    if not isinstance(r, u.Quantity) or \
       not isinstance(v, u.Quantity):
        raise TypeError("Position and Velocity must be Astropy "
                        "Quantity objects. You specified {0},{1}."
                        .format(type(r),type(v)))
    
    if not isiterable(r.value):
        r = u.Quantity([r.value], unit=r.unit)
    
    if not isiterable(v.value):
        v = u.Quantity([v.value], unit=v.unit)
    
    if r.value.shape != v.value.shape:
        raise ValueError("Shape mismatch: position and velocity "
                         "shapes must match. {0}, {1}"
                         .format(r.value.shape,v.value.shape))
    
    return r,v
    
class TestParticle(object):
    
    def __init__(self, r, v):
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
                
        r,v = _validate_r_v(r,v)
        self.r = r
        self.v = v
    
    def __getitem__(self, val):
        """ Implement indexing / slicing """
        
        if isinstance(val, int) or isinstance(val, slice) or isinstance(val, np.ndarray):
            return TestParticle(r=self.r[val], v=self.v[val])
        else:
            raise TypeError("Indices must be integer, not {0}".format(type(val)))
    
    def __len__(self):
        return len(self.r)
    
    def integrate(self, potential, t, integrator=leapfrog):
        """ Integrate the particle(s) from time t1 to t2 in the given 
            potential. 
            
            Parameters
            ----------
            potential : streams.Potential
                A Potential object to integrate the particles under.
            t : astropy.units.Quantity
                An array of times to integrate the particles on.
            integrator : func (optional)
                A function to use for integrating the particle orbits.
        """
        
        if not isinstance(potential, CartesianPotential):
            raise TypeError("potential must be a Potential object.")
        
        # test particles -- ignore particle-particle effects
        _t, r, v = integrator(potential.acceleration_at, 
                              self.r.decompose(bases=potential.units.values()).value,
                              self.v.decompose(bases=potential.units.values()).value, 
                              t=t)
        assert (t == _t).all()
        return _t,r,v
    
    '''
    def acceleration_at(self, r):
        """ Compute the gravitational acceleration from a collection of 
            massive particle objects.
            
            Parameters
            ----------
            r : astropy.units.Quantity
                Position to compute the acceleration at.
        """
        from ..potential.common import _cartesian_point_mass_model
        f,df = _cartesian_point_mass_model(bases={"length":self.r.unit,
                                                  "time":self.v.unit.bases[1],
                                                  "mass":self.m.unit})
        
        acc = np.zeros_like(r)
        for p in self:
            acc += df(p.r,origin=np.array([0.,0.,0.])*self.r.unit,m=p.m)
        
        return -acc
    '''
    
    def energy(self, potential):
        """ Compute the sum of the kinetic energy and potential energy of the
            particle(s) in the given potential.
            
            Parameters
            ----------
            potential : streams.Potential
                A Potential object.
        """
        
        raise NotImplementedError()
    
    def __repr__(self):
        if self.r.value.ndim == 1:
            return "<Particle r={1} v={2}>".format(self.r, self.v)
        else:
            return "[" + ", ".join(["<Particle r={1} v={2}>".format(p.r,p.v) for p in self]) + "]"
    
class TestParticleOrbit(object):
        
    def __init__(self, t, r, v):
        """ Represents an Orbit of a test particle in 3D coordinates """
                
        r,v = _validate_r_v(r,v)
        self.r = r
        self.v = v
        
        if self.r.value.ndim == 2:
            self.Nparticles = 1
        else:
            self.Nparticles = self.r.value.shape[1]
        
        if not isinstance(t,u.Quantity):
            raise TypeError("Time must be an astropy.units.Quantity, not {0}."
                            .format(type(t)))
        
        if len(t) != len(r):
            raise ValueError("Time array must have same length as positions, "
                             "velocities. ({0}) vs ({1})".format(len(t),len(r)))
                
        self.t = t
    
    def interpolate(self, t):
        """ Interpolate the orbit onto the specified time grid. """
        
        raise NotImplementedError()
    
    def normalized_phase_space_distance(self, other, r_norm, v_norm):
        """ Compute the phase-space distance between two orbits at each time.
            If the time arrays are different, throw an error.
        """
        if not (self.t == other.t).all():
            raise NotImplementedError("Interpolation not yet supported. Time "
                                      "vectors must be aligned.")
        
        if self.Nparticles == other.Nparticles:
            r_term = self.r-other.r
            v_term = self.v-other.v
        elif self.Nparticles == 1 and other.Nparticles > 1:
            r_term = self.r[:,np.newaxis,:] - other.r
            v_term = self.v[:,np.newaxis,:] - other.v            
        elif other.Nparticles == 1 and self.Nparticles > 1:
            r_term = self.r - other.r[:,np.newaxis,:]
            v_term = self.v - other.v[:,np.newaxis,:]
        else:
            raise ValueError("Size mismatch: {0} vs {1}"
                             .format(self.r.value.shape, other.r.value.shape))
        
        dist = np.sqrt(np.sum((r_term/r_norm).decompose()**2 + (v_term/v_norm).decompose()**2, axis=2))
        return dist
            