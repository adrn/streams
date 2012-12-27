# coding: utf-8

from __future__ import division, print_function

""" SGR_CEN / ORP_CEN are data tables containing information about the center of the dwarf
    galaxy at every timestep. The columns are:
        t, dt, x, y, z, vx, vy, vz

    SGR_SNAP / ORP_SNAP is are tables of particle information with columns:
        m,x,y,z,vx,vy,vz,s,s,tub
    and I should *skip line 1*
"""

# Standard library
import os
import sys
import copy
import time

# Third-party
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

from streams.potential import *
from streams.simulation import Particle, ParticleSimulation
from streams.util import SGRData

def plot_particle_xyz_vs_t(time_data, position_data, axes=None):
    """ Plot 1D time series for a chunk of data. This will assume that the data
        has shape (Ntimesteps, Nparticles, Ndimensions).
    """

    ntimesteps, nparticles, ndim = position_data.shape
    if axes == None:
        fig, axes = plt.subplots(ndim, 1, figsize=(14,10))

    for dim_idx in range(ndim):
        for particle_idx in range(nparticles):
            axes[dim_idx].plot(time_data, position_data[:,particle_idx,dim_idx])

    return axes

try:
    num_stars = int(sys.argv[1])
except IndexError:
    num_stars = 10

# Define potential as 3-component, bulge-disk-halo model
disk_potential = MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
bulge_potential = HernquistPotential(M=3.4E10*u.solMass, c=0.7)
halo_potential = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value, q1=1.38, q2=1.0, qz=1.36, phi=1.692969, c=12.)
galaxy_potential = disk_potential + bulge_potential + halo_potential

# Read in Kathryn's simulated data
sgrdata = SGRData(num_stars=num_stars)
t1 = min(sgrdata.sgr_cen["t"])
t2 = max(sgrdata.sgr_cen["t"])
dt = sgrdata.sgr_cen["dt"]

# Initialize simulation
simulation = ParticleSimulation(potential=galaxy_potential)

for ii in range(len(sgrdata.sgr_snap["x"][:num_stars])):
    p = Particle(position=(sgrdata.sgr_snap["x"][ii], sgrdata.sgr_snap["y"][ii], sgrdata.sgr_snap["z"][ii]), # kpc
                 velocity=(sgrdata.sgr_snap["vx"][ii], sgrdata.sgr_snap["vy"][ii], sgrdata.sgr_snap["vz"][ii]), # kpc/Myr
                 mass=1.) # M_sol
    simulation.add_particle(p)

# The data in SGR_CEN is only printed every 25 steps!
ts, xs, vs = simulation.run(t1=t2, t2=t1, dt=-dt)
ts = ts[::-25]
xs = xs[::-25,:,:]
vs = vs[::-25,:,:]

# Define tidal radius, escape velocity for satellite
msat = 2.5E8
sgr_cen_rs = np.sqrt(sgrdata.sgr_cen["x"]**2 + sgrdata.sgr_cen["y"]**2 + sgrdata.sgr_cen["z"]**2)
m_halo = halo_potential.params["v_halo"]**2*sgr_cen_rs/bulge_potential.params["_G"]
mr = disk_potential.params["M"] + bulge_potential.params["M"] + m_halo
r_tides = sgr_cen_rs * (msat / mr)**(1./3)
v_escs = np.sqrt(bulge_potential.params["_G"] * msat / r_tides)

# Escape velocity, distance per particle, looped over each timestep
energy_dists = np.ones(xs.shape[:2])
capture_times = 1E6*np.ones(xs.shape[1])
captured = np.zeros(xs.shape[1], dtype=int)
for ii in range(len(ts)):
    # Distance to satellite center and total velocity
    d = np.sqrt((xs[ii,:,0] - sgrdata.sgr_cen["x"][ii])**2 + (xs[ii,:,1] - sgrdata.sgr_cen["y"][ii])**2 + (xs[ii,:,2] - sgrdata.sgr_cen["z"][ii])**2)
    v = np.sqrt((vs[ii,:,0] - sgrdata.sgr_cen["vx"][ii])**2 + (vs[ii,:,1] - sgrdata.sgr_cen["vy"][ii])**2 + (vs[ii,:,2] - sgrdata.sgr_cen["vz"][ii])**2)
    energy_dists[ii] = np.sqrt((d/r_tides[ii])**2 + (v/v_escs[ii])**2)

    idx = (d < 1.8*r_tides[ii]) & (v < v_escs[ii]) & np.logical_not(captured.astype(bool))
    captured[idx] = 1
    capture_times[idx] = ts[ii]

print(sum(captured))
plt.figure()
delta_t = np.fabs(capture_times - sgrdata.sgr_snap["tub"])
plt.hist(delta_t[capture_times<1E6], bins=30)
plt.show()
sys.exit(0)

# Plot distance from Sgr center, velocity-v_esc
'''
fig, axes = plt.subplots(2, 1, figsize=(14,10))
axes[0].plot(ts, r_tides, color='m', linewidth=2, linestyle="--")
axes[1].plot(ts, v_escs, color='m', linewidth=2, linestyle="--")

for ii in range(num_stars):
    line = axes[0].plot(ts, np.sqrt((xs[:,ii,0]-sgrdata.sgr_cen["x"])**2 + \
                             (xs[:,ii,1]-sgrdata.sgr_cen["y"])**2 + \
                             (xs[:,ii,2]-sgrdata.sgr_cen["z"])**2))
    axes[0].axvline(sgrdata.sgr_snap["tub"][ii], color=line[0].get_color(), linewidth=2, linestyle='-')

    line = axes[1].plot(ts, np.sqrt((vs[:,ii,0]-sgrdata.sgr_cen["vx"])**2 + \
                             (vs[:,ii,1]-sgrdata.sgr_cen["vy"])**2 + \
                             (vs[:,ii,2]-sgrdata.sgr_cen["vz"])**2))
    axes[1].axvline(sgrdata.sgr_snap["tub"][ii], color=line[0].get_color(), linewidth=2, linestyle='-')

axes[0].set_ylabel("Dist. from Sgr Cen.")
axes[1].set_xlabel("Time [Myr]")
axes[1].set_ylabel("Velocity diff. from Sgr Cen.")
'''

min_energy_dists = energy_dists.argmin(axis=0)
plt.hist(min_energy_dists, bins=25)
plt.show()

sys.exit(0)

#########################################################

min_indices = particle_dists.argmin(axis=0)
min_times = ts[min_indices]
min_particle_dists = np.min(particle_dists, axis=0)

fig, axes = plt.subplots(2,2, figsize=(12,12), sharex=True, sharey=True)
axes[0,1].set_visible(False)
axes[0,0].set_xlim(-60, 60)
axes[0,0].set_ylim(-60, 60)
fig.subplots_adjust(hspace=0.0, wspace=0.0)

# Create circles to represent the tidal radius of the satellite
x0, y0, z0 = sgrdata.satellite_center["x"][0],sgrdata.satellite_center["z"][0],sgrdata.satellite_center["z"][0]
cenXY = plt.Circle((x0, y0), r_tides[0], facecolor='none',
            edgecolor='r', linewidth=2, alpha=0.75)
axes[0,0].add_patch(cenXY)
cenXZ = plt.Circle((x0, z0), r_tides[0], facecolor='none',
            edgecolor='r', linewidth=2, alpha=0.75)
axes[1,0].add_patch(cenXZ)
cenYZ = plt.Circle((y0, z0), r_tides[0], facecolor='none',
            edgecolor='r', linewidth=2, alpha=0.75)
axes[1,1].add_patch(cenYZ)

# Create circles for the stars
circlesXY = axes[0,0].scatter(xs[0,:num,0], xs[0,:num,1], marker='o', s=2., c='k', alpha=0.5, zorder=100)
circlesXZ = axes[1,0].scatter(xs[0,:num,0], xs[0,:num,2], marker='o', s=2., c='k', alpha=0.5, zorder=100)
circlesYZ = axes[1,1].scatter(xs[0,:num,1], xs[0,:num,2], marker='o', s=2., c='k', alpha=0.5, zorder=100)

# Draw the potential
grid = np.linspace(-60., 60., 100)
galaxy_potential.plot(grid, grid, grid, axes=axes)

for ii,t in enumerate(ts):
    circlesXY.set_offsets(xs[ii, :num, :2])
    circlesXZ.set_offsets(np.vstack((xs[ii, :num, 0], xs[ii, :num, 2])).T)
    circlesYZ.set_offsets(xs[ii, :num, 1:])

    cenXY.center = (sgrdata.satellite_center["x"][ii],sgrdata.satellite_center["y"][ii])
    cenXY.set_radius(r_tides[ii])
    cenXZ.center = (sgrdata.satellite_center["x"][ii],sgrdata.satellite_center["z"][ii])
    cenXZ.set_radius(r_tides[ii])
    cenYZ.center = (sgrdata.satellite_center["y"][ii],sgrdata.satellite_center["z"][ii])
    cenYZ.set_radius(r_tides[ii])

    #circles.set_facecolors(colors)
    plt.draw()
    #time.sleep(0.01)
    plt.savefig("plots/sgr/sgr_{0:03d}.png".format(ii))

sys.exit(0)

pos_units = "kpc"
# Position plots
fig, axes = plt.subplots(2,2,sharex=True, sharey=True, figsize=(12,12))
axes[0,0].scatter(xs[0,:,0], xs[0,:,1], color='k', alpha=0.5, s=4)
axes[0,0].scatter(xs[-1,:,0], xs[-1,:,1], color='r', alpha=0.5, s=4)
axes[0,0].set_ylabel("y [{0}]".format(pos_units))

axes[0,1].set_visible(False)

axes[1,0].scatter(xs[0,:,0], xs[0,:,2], color='k', alpha=0.5, s=4)
axes[1,0].scatter(xs[-1,:,0], xs[-1,:,2], color='r', alpha=0.5, s=4)
axes[1,0].set_xlabel("x [{0}]".format(pos_units))
axes[1,0].set_ylabel("z [{0}]".format(pos_units))

axes[1,1].scatter(xs[0,:,1], xs[0,:,2], color='k', alpha=0.5, s=4)
axes[1,1].scatter(xs[-1,:,1], xs[-1,:,2], color='r', alpha=0.5, s=4)
axes[1,1].set_xlabel("y [{0}]".format(pos_units))

fig.subplots_adjust(hspace=0, wspace=0)

plt.show()





def diagnostic_figures():
    fig = plt.figure(figsize=(14,11))
    ax = fig.add_subplot(111)
    ax.plot(ts, particle_dists[:,0], 'k-')
    ax.axvline(min_times[0], color='k', linestyle="--", linewidth=2)
    ax.axvline(sgrdata.star_snapshot["tub"][0], color='k', linestyle="-.", linewidth=2)

    ax.plot(ts, particle_dists[:,1], 'b-')
    ax.axvline(min_times[1], color='b', linestyle="--", linewidth=2)
    ax.axvline(sgrdata.star_snapshot["tub"][1], color='b', linestyle="-.", linewidth=2)

    plt.show()
    return

    plt.figure(figsize=(14,11))
    plt.subplot(311)
    plt.plot(ts, sgrdata.satellite_center["x"], 'r-', label="Center")
    plt.plot(ts, sgrdata.satellite_center["x"] - (cen_rs * (msat / mr)**(1./3)), 'r--')
    plt.plot(ts, sgrdata.satellite_center["x"] + (cen_rs * (msat / mr)**(1./3)), 'r--')
    plt.plot(ts, xs[:,0,0], 'k-', label="Star1")
    plt.axvline(min_times[0], color='k', linestyle="--", linewidth=3)
    plt.plot(ts, xs[:,1,0], 'b-', label="Star2")
    plt.axvline(min_times[1], color='b', linestyle="--", linewidth=3)
    plt.ylabel("x")
    plt.legend(loc="upper left",prop={'size':12})

    plt.subplot(312)
    plt.plot(ts, sgrdata.satellite_center["y"], 'r-')
    plt.plot(ts, sgrdata.satellite_center["y"] - (cen_rs * (msat / mr)**(1./3)), 'r--')
    plt.plot(ts, sgrdata.satellite_center["y"] + (cen_rs * (msat / mr)**(1./3)), 'r--')
    plt.plot(ts, xs[:,0,1], 'k-')
    plt.axvline(min_times[0], color='k', linestyle="--", linewidth=3)
    plt.plot(ts, xs[:,1,1], 'b-')
    plt.axvline(min_times[1], color='b', linestyle="--", linewidth=3)
    plt.ylabel("y")

    plt.subplot(313)
    plt.plot(ts, sgrdata.satellite_center["z"], 'r-')
    plt.plot(ts, sgrdata.satellite_center["z"] - (cen_rs * (msat / mr)**(1./3)), 'r--')
    plt.plot(ts, sgrdata.satellite_center["z"] + (cen_rs * (msat / mr)**(1./3)), 'r--')
    plt.plot(ts, xs[:,0,2], 'k-')
    plt.axvline(min_times[0], color='k', linestyle="--", linewidth=3)
    plt.plot(ts, xs[:,1,2], 'b-')
    plt.axvline(min_times[1], color='b', linestyle="--", linewidth=3)
    plt.xlabel("t")
    plt.ylabel("z")

    plt.show()
    return

    plt.subplot(211)
    plt.hist(min_particle_dists, bins=50)

    plt.subplot(212)
    plt.hist(min_times[min_times<0], bins=50)
    plt.show()

    return