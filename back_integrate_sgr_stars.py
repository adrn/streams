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
from streams.integrate import leapfrog
from streams.simulation import Particle, ParticleSimulation

def back_integrate_stars(num=1000):
    # Define potential as 3-component, bulge-disk-halo model
    disk_potential = MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
    bulge_potential = HernquistPotential(M=3.4E10*u.solMass, c=0.7)
    halo_potential = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value, q1=1.38, q2=1.0, qz=1.36, phi=1.692969, c=12.)
    galaxy_potential = disk_potential + bulge_potential + halo_potential

    # Read in Kathryn's simulated data
    sgrdata = SGRData(num_stars=num)
    t1, t2 = (min(sgrdata.satellite_center["t"]), max(sgrdata.satellite_center["t"]))
    dt = sgrdata.satellite_center["dt"]

    # Initialize simulation
    simulation = ParticleSimulation(potential=galaxy_potential)

    for ii in range(len(sgrdata.star_snapshot["x"])):
        p = Particle(position=(sgrdata.star_snapshot["x"][ii], sgrdata.star_snapshot["y"][ii], sgrdata.star_snapshot["z"][ii]), # kpc
                     velocity=(sgrdata.star_snapshot["vx"][ii], sgrdata.star_snapshot["vy"][ii], sgrdata.star_snapshot["vz"][ii]), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)

    # The data in SGR_CEN is only printed every 25 steps!
    ts, xs, vs = simulation.run(t1=t2, t2=t1, dt=-dt)
    ts = ts[::-25]
    xs = xs[::-25,:,:]
    vs = vs[::-25,:,:]

    # Define tidal radius, escape velocity for satellite
    msat = 2.5E8
    cen_rs = np.sqrt(sgrdata.satellite_center["x"]**2 + sgrdata.satellite_center["y"]**2 + sgrdata.satellite_center["z"]**2)
    m_halo = halo_potential.params["v_halo"]**2*cen_rs/bulge_potential.params["_G"]
    mr = disk_potential.params["M"] + bulge_potential.params["M"] + m_halo
    r_tides = (cen_rs * (msat / mr)**(1./3))
    v_escs = np.sqrt(2 * bulge_potential.params["_G"] * msat / r_tides)

    # Escape velocity, distance per particle, looped over each timestep
    #particle_dists = np.ones(xs.shape[:2])
    captured = np.zeros(xs.shape[1], dtype=int)
    for ii in range(len(ts)):
        # Distance to satellite center and total velocity
        d = np.sqrt((xs[ii,:,0] - sgrdata.satellite_center["x"][ii])**2 + (xs[ii,:,1] - sgrdata.satellite_center["y"][ii])**2 + (xs[ii,:,2] - sgrdata.satellite_center["z"][ii])**2)
        v = np.sqrt(vs[ii,:,0]**2 + vs[ii,:,1]**2 + vs[ii,:,2]**2)
        #particle_dists[ii] = np.sqrt((d/r_tide)**2 + (v/v_esc)**2)

        idx = (d < r_tides[ii]) & (v < v_escs[ii]) & np.logical_not(captured.astype(bool))
        captured[idx] = 1

    print(sum(captured))
    return

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

    return

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


def main():
    # Read in Kathryn's simulated data of the Sgr Dwarf center position / velocity
    sgrself._cen_data = ascii.read("data/SGR_CEN", data_start=1, names=["t", "dt", "x", "y", "z", "vx", "vy", "vz"])

    # Scalings to bring to physical units
    ru = 0.63
    vu = (41.27781037*u.km/u.s).to(u.kpc/u.Myr).value
    tu = 0.0149238134129*1000.

    sgr_x, sgr_y, sgr_z = np.array(ru*sgr_cen_data["x"]), np.array(ru*sgr_cen_data["y"]), np.array(ru*sgr_cen_data["z"])
    sgr_vx, sgr_vy, sgr_vz = np.array(vu*sgr_cen_data["vx"]), np.array(vu*sgr_cen_data["vy"]), np.array(vu*sgr_cen_data["vz"])
    sgr_t, dt = tu*sgr_cen_data["t"], tu*sgr_cen_data["dt"][0]

    print("r {0} kpc".format(np.sqrt(sgr_x[0]**2 + sgr_y[0]**2 + sgr_z[0]**2)))
    print("v {0} km/s".format( (np.sqrt(sgr_vx[0]**2 + sgr_vy[0]**2 + sgr_vz[0]**2)*u.kpc/u.Myr).to(u.km/u.s).value))

    t1, t2 = (min(sgr_t), max(sgr_t))

    disk_potential = MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
    bulge_potential = HernquistPotential(M=3.4E10*u.solMass, c=0.7)
    halo_potential = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value, q1=1.38, q2=1.0, qz=1.36, phi=1.692969, c=12.)
    galaxy_potential = disk_potential + bulge_potential + halo_potential

    initial_position = [sgr_x[0], sgr_y[0], sgr_z[0]]
    initial_velocity = [sgr_vx[0], sgr_vy[0], sgr_vz[0]]
    ts, xs, vs = leapfrog(galaxy_potential.acceleration_at, initial_position, initial_velocity, t1, t2, dt)

    # Compute energy and plot
    E_kin = 0.5*np.sum(vs**2, axis=2)
    E_pot = galaxy_potential.value_at(xs)

    energies = galaxy_potential.energy_at(xs, vs)
    delta_E = (energies - energies[0]) / energies[0]
    print(sum(delta_E))
    plt.figure()
    plt.semilogy(ts, delta_E, 'k.')
    plt.ylabel(r"$\Delta E/E$")
    plt.show()

    """
    fig, axes = plt.subplots(3,1,sharex=True,sharey=True)
    axes[0].plot(ts, E_kin)
    axes[0].set_ylabel(r"$E_{kin}$")

    axes[1].plot(ts, E_pot.T)
    axes[1].set_ylabel(r"$E_{pot}$")

    axes[2].plot(ts, delta_E, 'k.')
    axes[2].set_ylabel(r"$\Delta E/E$")

    plt.show()
    sys.exit(0)
    """

    pos_units = "kpc"
    time_units = "Myr"

    # Radius vs. time
    plt.figure()
    plt.plot(ts, np.sqrt(xs[:,0,0]**2+xs[:,0,1]**2+xs[:,0,2]**2), 'k-')
    plt.plot(sgr_t, np.sqrt(sgr_x**2 + sgr_y**2 + sgr_z**2), 'b-')

    # Phase-space plots
    fig2, axes2 = plt.subplots(3, 1, figsize=(14,10))
    axes2[0].plot(xs[:,0,0], vs[:,0,0], color='k')
    axes2[0].plot(sgr_x, sgr_vx, color='b')
    axes2[0].set_xlabel("x [{0}]".format(pos_units))
    axes2[0].set_ylabel(r"$v_x$ [{0}/{1}]".format(pos_units, time_units))

    axes2[1].plot(xs[:,0,1], vs[:,0,1])
    axes2[1].plot(sgr_y, sgr_vy, color='b')
    axes2[1].set_xlabel("y [{0}]".format(pos_units))
    axes2[1].set_ylabel(r"$v_y$ [{0}/{1}]".format(pos_units, time_units))

    axes2[2].plot(xs[:,0,2], vs[:,0,2])
    axes2[2].plot(sgr_z, sgr_vz, color='b')
    axes2[2].set_xlabel("z [{0}]".format(pos_units))
    axes2[2].set_ylabel(r"$v_z$ [{0}/{1}]".format(pos_units, time_units))

    # Position vs. Time
    fig3, axes3 = plt.subplots(3, 1, figsize=(14,10))
    axes3[0].plot(ts, xs[:,0,0])
    axes3[0].plot(sgr_t, sgr_x, 'r-', alpha=0.5, color='b')
    axes3[0].set_ylabel("x [{0}]".format(pos_units))

    axes3[1].plot(ts, xs[:,0,1])
    axes3[1].plot(sgr_t, sgr_y, 'r-', alpha=0.5, color='b')
    axes3[1].set_ylabel("y [{0}]".format("kpc"))

    axes3[2].plot(ts, xs[:,0,2])
    axes3[2].plot(sgr_t, sgr_z, 'r-', alpha=0.5, color='b')
    axes3[2].set_ylabel("z [{0}]".format("kpc"))
    axes3[2].set_xlabel("time [{0}]".format("Myr"))

    # Position plots
    fig, axes = plt.subplots(2,2,sharex=True, sharey=True, figsize=(12,12))
    axes[0,0].plot(xs[:,0,0], xs[:,0,1], linewidth=0.5, alpha=0.6)
    axes[0,0].plot(sgr_x, sgr_y, alpha=0.5, color='b')
    axes[0,0].set_ylabel("y [{0}]".format(pos_units))
    axes[0,0].plot(xs[0,0,0], xs[0,0,1], color='r', marker='o')

    axes[0,1].set_visible(False)

    axes[1,0].plot(xs[:,0,0], xs[:,0,2], linewidth=0.5, alpha=0.6)
    axes[1,0].plot(sgr_x, sgr_z, alpha=0.5, color='b')
    axes[1,0].set_xlabel("x [{0}]".format(pos_units))
    axes[1,0].set_ylabel("z [{0}]".format(pos_units))
    axes[1,0].plot(xs[0,0,0], xs[0,0,2], color='r', marker='o')

    axes[1,1].plot(xs[:,0,1], xs[:,0,2], linewidth=0.5, alpha=0.6)
    axes[1,1].plot(sgr_y, sgr_z, alpha=0.5, color='b')
    axes[1,1].plot(xs[0,0,1], xs[0,0,2], color='r', marker='o')
    axes[1,1].set_xlabel("y [{0}]".format(pos_units))

    fig.subplots_adjust(hspace=0, wspace=0)

    plt.show()

    #make_plots(ts, xs, vs, pos_units=u.kpc, time_units=u.Myr)

    return

    print("Done integrating!")
    plt.figure()
    plt.plot(ts, xs[:,0,0], 'k-', alpha=0.5, label='me')
    plt.plot(sgr_cen_data["t"], sgr_cen_data["x"], 'r-', alpha=0.5, label='kathryn')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #main()
    back_integrate_stars(5)