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
from scipy import interpolate

from ..potential import CartesianPotential
from ..integrate import leapfrog
from ..plot.data import scatter_plot_matrix

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
        
        #if not isinstance(potential, CartesianPotential):
        #    raise TypeError("potential must be a Potential object.")
        
        # THIS IS A HACK HACK HACK BECAUSE OF:
        #   https://github.com/astropy/astropy/issues/974
        r = u.Quantity(self.r.value, str(self.r.unit))
        v = u.Quantity(self.v.value, str(self.v.unit))
        t = u.Quantity(t.value, str(t.unit))
        
        # test particles -- ignore particle-particle effects
        _t, r, v = integrator(potential.acceleration_at, 
                              r.to(potential.units["length"]).value,
                              v.decompose(bases=potential.units.values()).value, 
                              t=t.decompose(bases=potential.units.values()).value)
        
        return TestParticleOrbit(_t*potential.units["time"], 
                                 r*potential.units["length"],
                                 v*potential.units["length"]/potential.units["time"])
    
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
            return "[" + ", ".join(["<Particle r={0} v={1}>".format(p.r,p.v) for p in self]) + "]"
    
    def plot_positions(self, **kwargs):
        """ Make a scatter-plot of 3 projections of the positions of the 
            particles in Galactocentric XYZ coordinates.
        """   
        
        labels = [r"${0}_{{GC}}$ [{1}]".format(nm, self.r.unit)
                    for nm in ["X", "Y", "Z"]]
        
        fig,axes = scatter_plot_matrix(self.r.value.T, 
                                       labels=labels,
                                       **kwargs)
        return fig, axes
        
    def plot_velocities(self, **kwargs):
        """ Make a scatter-plot of 3 projections of the velocities of the 
            particles. 
        """
        
        labels = [r"${0}_{{GC}}$ [{1}]".format(nm, self.r_unit)
                    for nm in ["V^x", "V^y", "V^z"]]
        
        fig,axes = scatter_plot_matrix(self.v.value.T, 
                                       labels=labels,
                                       **kwargs)
        return fig, axes
    
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
        """ Interpolate the orbit onto the specified time grid. 
        
            Parameters
            ----------
            t : astropy.units.Quantity
                The new grid of times to interpolate on to.
        """
        
        if not isinstance(t, u.Quantity):
            raise TypeError("New time grid must be an Astropy Quantity object.")
        
        new_t = t.to(self.t.unit)
        
        new_r = np.zeros((len(new_t), 3))
        new_v = np.zeros((len(new_t), 3))
        for i in range(3):
            new_r[:,i] = interpolate.interp1d(self.t.value, self.r[:,i].value, kind='cubic')(new_t.value)
            new_v[:,i] = interpolate.interp1d(self.t.value, self.v[:,i].value, kind='cubic')(new_t.value)
        
        return TestParticleOrbit(new_t, new_r*self.r.unit, new_v*self.v.unit)