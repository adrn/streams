# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import numpy as np

# Project
import streams.io as io
from streams.coordinates import _gc_to_hel, _hel_to_gc
from streams.coordinates.frame import heliocentric
from streams.potential.lm10 import LawMajewski2010
from streams.integrate import LeapfrogIntegrator
import streams.inference as si
from streams.inference.back_integrate import back_integration_likelihood

nparticles = 16
potential = LawMajewski2010()
simulation = io.SgrSimulation("sgr_nfw/M2.5e+08", "SNAP113")
particles = simulation.particles(N=nparticles, expr="tub!=0")\
                      .to_frame(heliocentric)
satellite = simulation.satellite()\
                      .to_frame(heliocentric)

p_hel = particles._X.copy()
s_hel = satellite._X.copy()

p_gc = _hel_to_gc(p_hel)
s_gc = _hel_to_gc(s_hel)

gc = np.vstack((s_gc,p_gc)).copy()
acc = np.zeros_like(gc[:,:3])

times = []
for ii in range(10):
    a = time.time()
    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    np.array(gc[:,:3]), np.array(gc[:,3:]),
                                    args=(gc.shape[0], acc))

    t, rs, vs = integrator.run(t1=6200, t2=0, dt=-1)
    times.append(time.time()-a)

print(np.min(times), "seconds per integration")

times = []
for ii in range(10):
    a = time.time()
    back_integration_likelihood(6200, 0, -1, potential, p_hel, s_hel,
                                2.5e8, 0.01)
    times.append(time.time()-a)

print(np.min(times), "seconds per likelihood call")

_config = """
name: test
data_file: data/observed_particles/2.5e8_N1024.hdf5
nparticles: 4
particle_idx: [16, 194, 575, 702]

potential:
    class_name: LawMajewski2010
    parameters: [q1, qz, phi, v_halo]

particles:
    parameters: [d,mul,mub,vr]

satellite:
    parameters: [d,mul,mub,vr]
"""

config = io.read_config(_config)
model = si.StreamModel.from_config(config)
truths = model.truths

times = []
for ii in range(10):
    a = time.time()
    model(truths)
    times.append(time.time()-a)

print(np.min(times), "seconds per model call")