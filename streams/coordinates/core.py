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

__all__ = ["gsr_to_hel"]

def gsr_to_hel(l, b, v_gsr, v_sun_lsr=[10.,5.25,7.17]*u.km/u.s, v_circ=220*u.km/u.s):
    """ Convert a velocity from the Galactic standard of rest (GSR) to 
        heliocentric radial velocity. 
    """
    assert hasattr(v_gsr, 'unit')
    assert hasattr(l, 'radian')
    assert hasattr(b, 'radian')
    
    v_lsr = v_gsr - v_circ * sin(l.radian) * cos(b.radian)
    
    # velocity correction for Sun relative to LSR
    v_correct = v_sun_lsr[0]*cos(b.radian)*cos(l.radian) + \
                v_sun_lsr[1]*cos(b.radian)*sin(l.radian) + \
                v_sun_lsr[2]*sin(b.radian)
    v_hel = v_lsr - v_correct
    
    return v_hel