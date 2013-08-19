# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable

from ..dynamics import ParticleCollection
from .rrlyrae import rrl_M_V, rrl_V_minus_I
from .core import apparent_magnitude

__all__ = ["parallax_error", "proper_motion_error",  \
           "apparent_magnitude", "rr_lyrae_add_observational_uncertainties", \
           "add_uncertainties_to_particles", "V_to_G"]

def V_to_G(V, V_minus_I):
    """ Convert Johnson V to Gaia G-band.
        
        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """
    return V - 0.0257 - 0.0924*V_minus_I - 0.1623*V_minus_I**2 + 0.009*V_minus_I**3

def parallax_error(V, V_minus_I):
    """ Compute the estimated GAIA parallax error as a function of apparent 
        V-band magnitude and V-I color. All equations are taken from the GAIA
        science performance book:
    
            http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1 
    
        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """
    
    # GAIA G mag
    g = V - 0.0257 - 0.0924*V_minus_I- 0.1623*V_minus_I**2 + 0.0090*V_minus_I**3
    z = 10**(0.4*(g-15.)) 
    
    p = g < 12.
    if isiterable(V):
        if sum(p) > 0:
            z[p] = 10**(0.4*(12. - 15.))
    else:
        if p:
            z = 10**(0.4*(12. - 15.))
    
    # "end of mission parallax standard"
    # σπ [μas] = (9.3 + 658.1·z + 4.568·z^2)^(1/2) · [0.986 + (1 - 0.986) · (V-IC)]
    dp = np.sqrt(9.3 + 658.1*z + 4.568*z**2) * (0.986 + (1 - 0.986)*V_minus_I) * 1E-6 * u.arcsecond
    
    return dp

def proper_motion_error(V, V_minus_I):
    """ Compute the estimated GAIA proper motion error as a function of apparent 
        V-band magnitude and V-I color. All equations are taken from the GAIA
        science performance book:
    
            http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1 
    
        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """
    
    dp = parallax_error(V, V_minus_I)
    
    # assume 5 year baseline, µas/year
    dmu = dp/u.year
    
    # too optimistic: following suggests factor 2 more realistic
    #http://www.astro.utu.fi/~cflynn/galdyn/lecture10.html 
    # - and Sanjib suggests factor 0.526
    dmu = 0.526*dmu

    return dmu.to(u.radian/u.yr)

def rr_lyrae_add_observational_uncertainties(x,y,z,vx,vy,vz,**kwargs):
    """ Given 3D galactocentric position and velocity, transform to heliocentric
        coordinates, apply observational uncertainty estimates, then transform
        back to galactocentric frame.
        
        TODO: V-I color and metallicity should be *draws* from distributions.
        
        Parameters
        ----------
        x,y,z : astropy.units.Quantity
            Positions.
        vx,vy,vz : astropy.units.Quantity
            Velocities.
    """
    
    if not isinstance(x,u.Quantity) or \
       not isinstance(y,u.Quantity) or \
       not isinstance(z,u.Quantity):
        raise TypeError("Positions must be Astropy Quantity objects!")
    
    if not isinstance(vx,u.Quantity) or \
       not isinstance(vy,u.Quantity) or \
       not isinstance(vz,u.Quantity):
        raise TypeError("Velocities must be Astropy Quantity objects!")
    
    # assuming [Fe/H] = -0.5 for Sgr
    M_V, dM_V = rrl_M_V(-0.5)
    
    # Transform to heliocentric coordinates
    rsun = 8.*u.kpc
    
    x = x + rsun
    
    d = np.sqrt(x.value**2 + y.value**2 + z.value**2)*x.unit
    V = apparent_magnitude(M_V, d)
    
    vr = (x*vx + y*vy + z*vz) / d 
    
    # proper motions in km/s/kpc
    rad = np.sqrt(x.value**2 + y.value**2)*x.unit
    vrad = (x*vx + y*vy) / rad
    mul = (x*vy - y*vx) / rad / d
    mub = (-z*vrad + rad*vz) / d**2
       
    # angular position
    sinb = z/d
    cosb = rad/d
    cosl = x/rad
    sinl = y/rad
    
    # DISTANCE ERROR -- assuming 2% distances from RR Lyrae mid-IR
    if kwargs.has_key("distance_error_percent") and \
        kwargs["distance_error_percent"] is not None:
        d_err = kwargs["distance_error_percent"] / 100.
    else:
        d_err = 0.02
    d += np.random.normal(0., d_err*d.value)*d.unit
    
    # RADIAL VELOCITY ERROR -- 10 km/s
    if kwargs.has_key("radial_velocity_error") and \
        kwargs["radial_velocity_error"] is not None:
        rv_err = kwargs["radial_velocity_error"]
    else:
        rv_err = 10.*u.km/u.s
    
    rv_err = rv_err.to(u.km/u.s)
    vr += np.random.normal(0., rv_err.value, size=len(vr))*rv_err.unit
    
    # PROPER MOTION ERROR
    if kwargs.has_key("proper_motion_error") and \
        kwargs["proper_motion_error"] is not None:
        dmu = kwargs["proper_motion_error"]
    else:
        dmu = proper_motion_error(V, rrl_V_minus_I)            
    
    dmu = (dmu.to(u.rad/u.s).value / u.s).to(u.km / (u.kpc*u.s))
    try:
        size = len(dmu)
    except TypeError:
        size = len(vr)
    
    mul += np.random.normal(0., dmu.value, size=size)*dmu.unit
    mub += np.random.normal(0., dmu.value, size=size)*dmu.unit
    
    # compute tangential velocities as distance times proper motion
    v_l = d*mul
    v_b = d*mub
    
    new_x = (d*cosb*cosl - rsun).to(d.unit).value
    new_y = (d*cosb*sinl).to(d.unit).value
    new_z = (d*sinb).to(d.unit).value
    
    new_vx = (vr*cosb*cosl - v_l*sinl - v_b*sinb*cosl).to(vr.unit).value
    new_vy = (vr*cosb*sinl + v_l*cosl - v_b*sinb*sinl).to(vr.unit).value
    new_vz = (vr*sinb + v_b*cosb).to(vr.unit).value
    
    return (new_x*x.unit, new_y*x.unit, new_z*x.unit, 
            new_vx*vx.unit, new_vy*vx.unit, new_vz*vx.unit)

def add_uncertainties_to_particles(particles, **kwargs):
    """ Given a ParticleCollection object, add RR Lyrae-like uncertainties 
        and return a new ParticleCollection with the errors.
    """
    
    x,y,z,vx,vy,vz = rr_lyrae_add_observational_uncertainties(particles.r[:,0],
                                                              particles.r[:,1],
                                                              particles.r[:,2],
                                                              particles.v[:,0],
                                                              particles.v[:,1],
                                                              particles.v[:,2],
                                                              **kwargs)
    
    new_r = np.zeros_like(particles.r.value)
    new_r[:,0] = x.value
    new_r[:,1] = y.value
    new_r[:,2] = z.value
    
    new_v = np.zeros_like(particles.v.value)
    new_v[:,0] = vx.value
    new_v[:,1] = vy.value
    new_v[:,2] = vz.value
    
    return ParticleCollection(r=new_r*particles.r.unit, 
                              v=new_v*particles.v.unit,
                              m=particles.m,
                              unit_system=particles.unit_system)
