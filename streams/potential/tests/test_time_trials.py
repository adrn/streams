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

from ...misc.units import UnitSystem
from ..core import *
from ..common import *
from ..lm10 import LawMajewski2010
from ..pal5 import Palomar5

Ntrials = 100
Nparticles = 1000

usys = UnitSystem(u.kpc, u.M_sun, u.Myr, u.radian)

def time_potential(potential, *args):
    _r = np.random.uniform(1, 50, size=(Nparticles, 3))
    print()
    print(potential)
    
    a = time.time()
    for ii in range(Ntrials):
        potential._acceleration_at(_r, *args)
        
    t = (time.time()-a)/float(Ntrials)
    print("{0} particles, {1:.3f} ms per call".format(Nparticles, t*1E3))

def test_time_miyamoto():
           
    potential = MiyamotoNagaiPotential(unit_system=usys,
                                       m=1.E11*u.M_sun, 
                                       a=6.5*u.kpc,
                                       b=0.26*u.kpc,
                                       r_0=[0.,0.,0.]*u.kpc)
    time_potential(potential)

def test_time_hernquist():
           
    potential = HernquistPotential(unit_system=usys,
                                   m=1.E11*u.M_sun, 
                                   c=0.7*u.kpc)
    time_potential(potential)

def test_time_log():
           
    potential = LogarithmicPotentialLJ(unit_system=usys,
                                           q1=1.4,
                                           q2=1.,
                                           qz=1.5,
                                           phi=1.69*u.radian,
                                           v_halo=120.*u.km/u.s,
                                           r_halo=12.*u.kpc)
                                           
    time_potential(potential)

def test_time_composite():
    potential = CompositePotential(unit_system=usys)
    potential["disk"] = MiyamotoNagaiPotential(unit_system=usys,
                                       m=1.E11*u.M_sun, 
                                       a=6.5*u.kpc,
                                       b=0.26*u.kpc,
                                       r_0=[0.,0.,0.]*u.kpc)
    
    potential["bulge"] = HernquistPotential(unit_system=usys,
                                   m=1.E11*u.M_sun, 
                                   c=0.7*u.kpc)
    
    potential["halo"] = LogarithmicPotentialLJ(unit_system=usys,
                                       q1=1.4,
                                       q2=1.,
                                       qz=1.5,
                                       phi=1.69*u.radian,
                                       v_halo=120.*u.km/u.s,
                                       r_halo=12.*u.kpc)
    
    print("Pure-python LM10")
    time_potential(potential)

def test_lm10():
    print("LM10")
    py = LawMajewski2010(v_halo=121*u.km/u.s)
    acc = np.zeros((Nparticles,3))
    time_potential(py, Nparticles, acc)
    
def test_pal5():
    print("Pal5")
    py = Palomar5()
    acc = np.zeros((Nparticles,3))
    time_potential(py, Nparticles, acc)

