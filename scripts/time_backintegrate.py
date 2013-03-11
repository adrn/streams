# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

# Project
from streams.simulation import run_back_integration
from streams.data import SgrSnapshot, SgrCen
from streams.potential import *

sgr_cen = SgrCen()
sgr_snap = SgrSnapshot(num=100, no_bound=True)

# Get timestep information from SGR_CEN
t1 = min(sgr_cen.data["t"])
t2 = max(sgr_cen.data["t"])
dt = sgr_cen.data["dt"][0]*10

# Interpolate SgrCen data onto new times
ts = np.arange(t2, t1, -dt)
sgr_cen.interpolate(ts)

halo_params = dict(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value,
                   q1=1.38,
                   q2=1.0,
                   qz=1.36,
                   phi=1.692969,
                   c=12.)

#halo_potential = LogarithmicPotentialLJ(**halo_params)
#run_back_integration(halo_potential, sgr_snap, sgr_cen, dt=dt)

import time

times = []
for qz in np.linspace(0.5, 1.5, 100):
    a = time.time()
    halo_params["qz"] = qz
    halo_potential = LogarithmicPotentialLJ(**halo_params)
    run_back_integration(halo_potential, sgr_snap, sgr_cen, dt)
    times.append(time.time()-a)

print("Average time per back integration: {0}".format(np.mean(times)))