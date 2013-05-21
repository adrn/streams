# coding: utf-8
"""
    Time the acceleration function calls for the main galaxy 
    potential components
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import time
import pytest
import numpy as np
from astropy.constants.si import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import *
from ..common import *

Ntrials = 100
Nparticles = 100

usys = UnitSystem(u.kpc, u.M_sun, u.Myr, u.radian)

def time_potential(potential):
    r = np.random.uniform(1, 50, size=(Nparticles, 3))*u.kpc
    print()
    
    a = time.time()
    for ii in range(Ntrials):
        potential.acceleration_at(r)
        
    t = (time.time()-a)/float(Ntrials)
    print("With units: {0:.3f} ms per call".format(t*1E3))
    
    _r = r.value
    a = time.time()
    for ii in range(Ntrials):
        potential._acceleration_at(_r)
        
    t = (time.time()-a)/float(Ntrials)
    print("Without units: {0:.3f} ms per call".format(t*1E3))

def test_time_miyamoto():
           
    potential = MiyamotoNagaiPotential(unit_system=usys,
                                       m=1.E11*u.M_sun, 
                                       a=6.5*u.kpc,
                                       b=0.26*u.kpc,
                                       r_0=[0.,0.,0.]*u.kpc)
    time_potential(potential)

