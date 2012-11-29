# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import pytest
import numpy as np

from ..core import *
from ...potential import *

# API?:
disk_potential = MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
#bulge_potential = HernquistPotential(M=3.4E10*u.solMass, c=0.7)
#halo_potential = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value, q1=1.38, q2=1.0, qz=1.36, phi=1.692969, c=12.)
#galaxy_potential = disk_potential + bulge_potential + halo_potential
potential = disk_potential

simulation = ParticleSimulation(potential=potential)

for ii in range(100):
    p = Particle(position=(np.random.uniform(1., 10.), 0., np.random.uniform(-0.5, 0.5)), # kpc
                 velocity=((np.random.uniform(-10., 10.)*u.km/u.s).to(u.kpc/u.Myr).value, (200*u.km/u.s).to(u.kpc/u.Myr).value, (np.random.uniform(-10., 10.)*u.km/u.s).to(u.kpc/u.Myr).value), # kpc/Myr
                 mass=1.) # M_sol
    simulation.add_particle(p)

assert (simulation.particles[0].position == simulation._particle_pos_array[0]).all()
assert (simulation.particles[0].velocity == simulation._particle_vel_array[0]).all()
assert (simulation.particles[11].position == simulation._particle_pos_array[11]).all()
assert (simulation.particles[11].velocity == simulation._particle_vel_array[11]).all()

ts, xs, vs = simulation.run(t1=0., t2=1000., dt=1.0)

import matplotlib.pyplot as plt
plt.figure(1)
circles = plt.scatter(xs[0,:,0], xs[0,:,1], marker='o', s=1., c='k', alpha=0.2)
for ii,t in enumerate(ts):
    circles.set_offsets(xs[ii, :, :2])
    #circles.set_facecolors(colors)
    plt.draw()

#simulation.run(t1=0., t2=1000., dt=0.1)
# animation = simulation.animate() # Make snapshots of all timesteps, and make an animation

#ax = simulation.plot(t=1000.)
#plt.show()