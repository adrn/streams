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

# Third-party
import matplotlib.pyplot as plt
from astropy.io import ascii
import astropy.units as u
import numpy as np

from streams.potential import *
from streams.integrate import leapfrog

def main():
    # Read in Kathryn's simulated data of the Sgr Dwarf center position / velocity
    sgr_cen_data = ascii.read("data/SGR_CEN", data_start=1, names=["t", "dt", "x", "y", "z", "vx", "vy", "vz"])

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
    bulge_potential = HernquistPotential(M=3.37509E10*u.solMass, c=0.7)
    halo_potential = LogarithmicPotentialJHB(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value, d=12.)
    galaxy_potential = disk_potential + bulge_potential + halo_potential

    initial_position = [sgr_x[0], sgr_y[0], sgr_z[0]]
    initial_velocity = [sgr_vx[0], sgr_vy[0], sgr_vz[0]]
    ts, xs, vs = leapfrog(galaxy_potential.acceleration_at, initial_position, initial_velocity, t1, t2, dt)

    '''
    E_kin = 0.5*np.sum(vs**2, axis=2)
    E_pot = galaxy_potential.value_at(xs)

    energies = galaxy_potential.energy_at(xs, vs)

    fig, axes = plt.subplots(3,1,sharex=True,sharey=True)
    axes[0].plot(ts, E_kin)
    axes[0].set_ylabel(r"$E_{kin}$")

    axes[1].plot(ts, E_pot.T)
    axes[1].set_ylabel(r"$E_{pot}$")

    axes[2].plot(ts, (E_kin + E_pot.T), 'k.')
    axes[2].set_ylabel(r"$E_{tot}$")

    plt.show()
    sys.exit(0)
    '''

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
    axes3[2].set_xlabel("time [{0}/{1}]".format("kpc", "Myr"))

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
    main()