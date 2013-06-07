# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from ...potential.lm10 import LawMajewski2010
from ...data import lm10_particles, lm10_satellite, lm10_time
from ..satellite_particles import SatelliteParticleIntegrator

plot_path = "plots/tests/integrate/satellite_particles"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

lm10 = LawMajewski2010()
satellite = lm10_satellite()
particles = lm10_particles(N=100, expr="(Pcol>-1) & (abs(Lmflag)==1) & (dist < 80)")
t1,t2 = lm10_time()

def test_sp():
    integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
    integrator.run(time_spec=dict(t1=t1, t2=t2, dt=-5.))
    
    integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
    integrator.run(time_spec=dict(t1=t1, t2=t2),
                   timestep_func=lambda r,v: -5.)