# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

# Project
from streams.simulation import back_integrate, Particle, TestParticleSimulation, minimum_distance_matrix
from streams.data import SgrSnapshot, SgrCen
from streams.potential import *
from streams.potential.lm10 import halo_params as true_halo_params

np.random.seed(42)
sgr_cen = SgrCen()
sgr_snap = SgrSnapshot(num=1000)

# Get timestep information from SGR_CEN
t1 = min(sgr_cen.t)
t2 = max(sgr_cen.t)
dt = sgr_cen.dt[0]*10

# Interpolate SgrCen data onto new times
ts = np.arange(t2, t1, -dt)*u.Myr
sgr_cen.interpolate(ts)

# Define true potential
mw_potential = LawMajewski2010(**true_halo_params)

true_halo_params["qz"] = 1.0
wrong_potential = LawMajewski2010(**true_halo_params)

def run_sim(potential):
    # Initialize particle simulation with full potential
    simulation = TestParticleSimulation(potential=potential)
    
    # Distances in kpc, velocities in kpc/Myr
    xyz = sgr_snap.xyz
    vxyz = sgr_snap.vxyz
    
    for ii in range(len(sgr_snap)):
        p = Particle(position=(xyz[0,ii].to(u.kpc).value, xyz[1,ii].to(u.kpc).value, xyz[2,ii].to(u.kpc).value), # kpc
                     velocity=(vxyz[0,ii].to(u.kpc/u.Myr).value, vxyz[1,ii].to(u.kpc/u.Myr).value, vxyz[2,ii].to(u.kpc/u.Myr).value), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)
    
    ts, xs, vs = simulation.run(t1=max(sgr_cen.t), t2=min(sgr_cen.t), dt=-dt)
    min_ps = minimum_distance_matrix(potential, xs, vs, sgr_cen)
    
    return min_ps

min_ps_true = run_sim(mw_potential)
min_ps_wrong = run_sim(wrong_potential)

print(np.sum(min_ps_wrong**2, axis=1).shape)

n,bins,batches = plt.hist(np.sum(min_ps_wrong**2, axis=1), bins=25, color="r", histtype="step", alpha=1, linewidth=2)
n,bins,batches = plt.hist(np.sum(min_ps_true**2, axis=1), bins=bins, color="k", histtype="step", alpha=0.75, linewidth=2)
plt.ylim(0,300)
plt.savefig("plots/test.png")