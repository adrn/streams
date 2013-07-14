# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from numpy import cos, sin

import astropy.coordinates as coord
import astropy.units as u

__all__ = ["vgsr_to_vhel", "vhel_to_vgsr"]

def vgsr_to_vhel(l, b, v_gsr, 
                 v_sun_lsr=[10.,5.25,7.17]*u.km/u.s, 
                 v_circ=220*u.km/u.s):
    """ Convert a velocity from the Galactic standard of rest (GSR) to 
        heliocentric radial velocity. 
        
        Parameters
        ----------
        
    """
    
    try:
        v_lsr = v_gsr - v_circ * sin(l.radian) * cos(b.radian)
    except AttributeError:
        raise AttributeError("All inputs must be Quantity objects")
    
    # velocity correction for Sun relative to LSR
    v_correct = v_sun_lsr[0]*cos(b.radian)*cos(l.radian) + \
                v_sun_lsr[1]*cos(b.radian)*sin(l.radian) + \
                v_sun_lsr[2]*sin(b.radian)
    v_hel = v_lsr - v_correct
    
    return v_hel

def vhel_to_vgsr(l, b, v_hel, 
                 v_sun_lsr=[10.,5.25,7.17]*u.km/u.s, 
                 v_circ=220*u.km/u.s):
    """ Convert a velocity from a heliocentric radial velocity to
        the Galactic center of rest.
        
        Parameters
        ----------
        
    """
    try:
        v_lsr = v_hel + v_circ * sin(l.radian) * cos(b.radian)
    except AttributeError:
        raise AttributeError("All inputs must be Quantity objects")
    
    # velocity correction for Sun relative to LSR
    v_correct = v_sun_lsr[0]*cos(b.radian)*cos(l.radian) + \
                v_sun_lsr[1]*cos(b.radian)*sin(l.radian) + \
                v_sun_lsr[2]*sin(b.radian)
    v_gsr = v_lsr + v_correct
    
    return v_gsr

def __radial_velocity(r, v):
    """ Compute the radial velocity in the heliocentric frame. 
        
        DON'T USE
    
    """
    
    if r.ndim < 2:
        r = r[np.newaxis]
    
    if v.ndim < 2:
        v = v[np.newaxis]
    
    # the sun's velocity and position
    v_circ = 220.
    v_sun = np.array([[0., v_circ, 0]]) # km/s
    v_sun += np.array([[9, 11., 6.]]) # km/s
    r_sun = np.array([[-8., 0, 0]])
    
    # object's distance in relation to the sun(observed radius)
    r_rel = r - r_sun
    R_obs = np.sqrt(np.sum(r_rel**2, axis=-1))[:,np.newaxis]
    r_hat = r_rel / R_obs
    
    v_rel = v - v_sun
    v_hel = np.sum((v_rel*r_hat), axis=-1)[:,np.newaxis]
    
    return np.squeeze(v_hel)