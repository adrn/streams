# coding: utf-8

""" Turn our model into a generative model. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmin

# Project
import streams.io as io
from streams.coordinates import _gc_to_hel, _hel_to_gc
from streams.coordinates.frame import heliocentric
from streams.potential.lm10 import LawMajewski2010
from streams.integrate import LeapfrogIntegrator
from streams.integrate.stream_generator import StreamGenerator
from streams.util import project_root

plot_path = os.path.join(project_root, "plots")

potential = LawMajewski2010()
nparticles = 2000
T = 4000.
dt = 0.1
_m = "2.e4"
np.random.seed(42)

# Set up a bit to specify leading or trailing
tail_bit = np.ones(nparticles)
tail_bit[:nparticles//2] = -1.

mass = float(_m)
# simulation = io.SgrSimulation(mass=_m)
# particles = simulation.particles(N=1000, expr="tub!=0")
# satellite = simulation.satellite()\
#                       .to_frame(heliocentric)

# s_hel = satellite._X.copy()
# s_gc = _hel_to_gc(s_hel)
s_gc = np.array([[8.363919011, 0.243352771, 16.864546659,
                  -0.04468993,-0.12392801,-0.01664498]]) # Pal5
s_hel = _gc_to_hel(s_gc)

# First integrate the orbit of the satellite back to get initial conditions
acc = np.zeros_like(s_gc[:,:3])
integrator = LeapfrogIntegrator(potential._acceleration_at,
                                np.array(s_gc[:,:3]), np.array(s_gc[:,3:]),
                                args=(s_gc.shape[0], acc))

t, rs, vs = integrator.run(t1=T, t2=0, dt=-dt)
init_r,init_v = rs[-1], vs[-1]

# integrate the orbit of the satellite
acc = np.zeros_like(s_gc[:,:3])
integrator = LeapfrogIntegrator(potential._acceleration_at,
                                init_r, init_v,
                                args=(1, acc))

t, rs, vs = integrator.run(t1=0, t2=T, dt=dt)

satellite_orbit = np.vstack((rs.T,vs.T)).T

# sample unbinding times uniformly
s_R_orbit = np.sqrt(np.sum(satellite_orbit[...,:3]**2, axis=-1))
pericenters, = argrelmin(np.squeeze(s_R_orbit))
pericenters = pericenters[:-1]

############################################
ppp = s_R_orbit[pericenters,0]
zero_one = (ppp - ppp.min()) / (ppp.max() - ppp.min())
#ppp = ((-zero_one + 1.)*99 + 1).astype(int)

ppp = zero_one**(0.3333333)
zero_one = (ppp - ppp.min()) / (ppp.max() - ppp.min())
ppp = ((-zero_one + 1.)*99 + 1).astype(int)
pp = []
for ii,peri in enumerate(pericenters):
    pp += [peri]*ppp[ii]

import random
tubs = []
for ii in range(nparticles):
    peri_idx = random.choice(pp)
    tub = np.random.normal(peri_idx+100, 100)
    tubs.append(tub)
tubs = np.array(tubs).astype(int)

acc = np.zeros((nparticles,3))
gen = StreamGenerator(potential, satellite_orbit, mass,
                      acc_args=(nparticles, acc))

t, orbits = gen.run(tubs, 0., T, dt)

# E_pot = potential._value_at(orbits[:,0,:3])
# E_kin = np.squeeze(0.5*np.sum(orbits[:,0,3:]**2, axis=-1))
# E_total = (E_pot + E_kin)[tubs[0]:]

# plt.clf()
# plt.subplot(211)
# plt.semilogy(np.abs(E_pot))
# plt.subplot(212)
# plt.semilogy(np.abs(E_kin))
# plt.savefig(os.path.join(plot_path, "wtf2.png"))

# plt.clf()
# plt.semilogy(np.fabs((E_total[1:]-E_total[0])/E_total[0]))
# plt.savefig(os.path.join(plot_path, "wtf.png"))
# sys.exit(0)

plt.clf()
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)

ax.plot(orbits[-1,:,0], orbits[-1,:,1],
             alpha=0.75, linestyle='none')
ax.plot(satellite_orbit[:,0,0], satellite_orbit[:,0,1],
             alpha=0.75, marker=None)
#axes[0].set_xlim(-80,40)
#axes[0].set_ylim(-60,60)
fig.savefig(os.path.join(plot_path, "generate_xy.png"))

plt.clf()
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)

hel = _gc_to_hel(orbits[-1])

l,b = hel[:,0], hel[:,1]
l = coord.Angle(l*u.rad).to(u.degree).wrap_at(180*u.deg).value
b = coord.Angle(b*u.rad).to(u.degree).value
ax.plot(l, b,
        alpha=0.75, linestyle='none')

l = coord.Angle(s_hel[:,0]*u.rad).to(u.degree).wrap_at(180*u.deg).value
b = coord.Angle(s_hel[:,1]*u.rad).to(u.degree).value
ax.plot(l,b)

# ax.plot(satellite_orbit[:,0,0], satellite_orbit[:,0,1],
#              alpha=0.75, marker=None)
#axes[0].set_xlim(-80,40)
#axes[0].set_ylim(-60,60)
fig.savefig(os.path.join(plot_path, "generate_lb.png"))


sys.exit(0)








########################################################
########################################################
########################################################

plt.clf()
fig,axes = plt.subplots(1, 2, figsize=(12,6),
                        sharex=True, sharey=True)
axes[0].plot(particles._X[:,0], particles._X[:,2],
             alpha=0.75, linestyle='none')
axes[1].plot(orbits[-1,:,0], orbits[-1,:,2],
             alpha=0.75, linestyle='none')
axes[0].set_xlim(-80,40)
axes[0].set_ylim(-60,60)
fig.savefig(os.path.join(plot_path, "generate2.png"))
sys.exit(0)

plt.clf()
plt.plot(particles._X[:,0], particles._X[:,2],
         alpha=0.5, linestyle='none')

orbits = np.zeros((T,nparticles,6))
for ii,tub in enumerate(tubs):
    print(ii)
    init_r = np.random.normal(rs[tub] + a_pm[tub,ii]*r_tide[tub], r_tide[tub])
    init_v = np.random.normal(vs[tub], v_disp[tub])

    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    init_r, init_v,
                                    args=(1, np.zeros_like(init_r)))
    t, R, V = integrator.run(t1=tub, t2=T, dt=dt)

    # orbits[tub:,ii,:3] = R[:,0]
    # orbits[tub:,ii,3:] = V[:,0]

    plt.plot(R[-1,0,0], R[-1,0,2], marker='.', color='b', alpha=0.5)

plt.savefig("/Users/adrian/projects/streams/plots/generate.png")