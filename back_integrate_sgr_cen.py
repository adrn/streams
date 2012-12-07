# coding: utf-8

""" Code for integrating the Sagittarius Dwarf *center* through a given potential. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# streams
from streams.util import SGRData
from streams.potential import *
from streams.integrate import leapfrog

# Read in data from Kathryn's simulation
sgr_data = SGRData()

# Define potential as used by Kathryn's simulation, to make sure I can recover the correct behavior
disk_potential = MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
bulge_potential = HernquistPotential(M=3.4E10*u.solMass, c=0.7)
halo_potential = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value, q1=1.38, q2=1.0, qz=1.36, phi=1.692969, c=12.)
galaxy_potential = disk_potential + bulge_potential + halo_potential

# ------------------------------------------------------------------------------
# Take the first timestep from SGR_CEN, integrate those conditions forward
# ------------------------------------------------------------------------------

# Read initial conditions / timesteps from data
initial_position = [sgr_data.sgr_cen["x"][0], sgr_data.sgr_cen["y"][0], sgr_data.sgr_cen["z"][0]]
initial_velocity = [sgr_data.sgr_cen["vx"][0], sgr_data.sgr_cen["vy"][0], sgr_data.sgr_cen["vz"][0]]
t1 = min(sgr_data.sgr_cen["t"])
t2 = max(sgr_data.sgr_cen["t"])

# Integrate the satellite center forward in the potential
ts_forward, xs_forward, vs_forward = leapfrog(galaxy_potential.acceleration_at, initial_position, initial_velocity, t1, t2, sgr_data.sgr_cen["dt"])

# Compute energies for this run
energies_forward = galaxy_potential.energy_at(xs_forward, vs_forward)
delta_E_forward = (energies_forward - energies_forward[0]) / energies_forward[0]

# -------------------------------------------------------------------------------
# Now take the last timestep from SGR_CEN, integrate those conditions backward
# -------------------------------------------------------------------------------

initial_position = [sgr_data.sgr_cen["x"][-1], sgr_data.sgr_cen["y"][-1], sgr_data.sgr_cen["z"][-1]]
initial_velocity = [sgr_data.sgr_cen["vx"][-1], sgr_data.sgr_cen["vy"][-1], sgr_data.sgr_cen["vz"][-1]]
t1 = max(sgr_data.sgr_cen["t"])
t2 = min(sgr_data.sgr_cen["t"])

# Integrate the satellite center backward in the potential from the last timestep data
ts_back, xs_back, vs_back = leapfrog(galaxy_potential.acceleration_at, initial_position, initial_velocity, t1, t2, -sgr_data.sgr_cen["dt"])

# Compute energies for this run
energies_back = galaxy_potential.energy_at(xs_back, vs_back)
delta_E_back = (energies_back - energies_back[0]) / energies_back[0]

fig = plt.figure(figsize=(15,12))

ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

# Radius from galactic center vs. time
ax1.plot(ts_forward, np.sqrt(xs_forward[:,0,0]**2+xs_forward[:,0,1]**2+xs_forward[:,0,2]**2), color='r', alpha=0.75, label="forward")
ax1.plot(ts_back, np.sqrt(xs_back[:,0,0]**2+xs_back[:,0,1]**2+xs_back[:,0,2]**2), color='b', alpha=0.75, label="back")
ax1.plot(sgr_data.sgr_cen["t"], np.sqrt(sgr_data.sgr_cen["x"]**2+sgr_data.sgr_cen["y"]**2+sgr_data.sgr_cen["z"]**2), color='k', alpha=0.75, linewidth=2)

# Phase space plots |r| vs. |v|
ax2.plot(np.sqrt(xs_forward[:,0,0]**2+xs_forward[:,0,1]**2+xs_forward[:,0,2]**2), np.sqrt(vs_forward[:,0,0]**2+vs_forward[:,0,1]**2+vs_forward[:,0,2]**2), color='r', alpha=0.75, label="forward")
ax2.plot(np.sqrt(xs_back[:,0,0]**2+xs_back[:,0,1]**2+xs_back[:,0,2]**2), np.sqrt(vs_back[:,0,0]**2+vs_back[:,0,1]**2+vs_back[:,0,2]**2), color='b', alpha=0.75, label="back")
ax2.plot(np.sqrt(sgr_data.sgr_cen["x"]**2+sgr_data.sgr_cen["y"]**2+sgr_data.sgr_cen["z"]**2), np.sqrt(sgr_data.sgr_cen["vx"]**2+sgr_data.sgr_cen["vy"]**2+sgr_data.sgr_cen["vz"]**2), color='k', alpha=0.75, linewidth=2)

# Energy
ax3.plot(ts_forward, delta_E_forward, color='r', alpha=0.75, label="forward")
ax3.plot(ts_back, delta_E_back, color='b', alpha=0.75, label="back")
ax3.legend()

plt.show()