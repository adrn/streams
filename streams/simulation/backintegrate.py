# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

from streams.potential import *
from streams.simulation import Particle, ParticleSimulation

def run_back_integration(halo_potential, sgr_snap, sgr_cen):
    """ Given the particle snapshot information and a potential, integrate the particles
        backwards and return the minimum energy distances.
    """

    # Get timestep information from SGR_CEN
    t1 = min(sgr_cen.data["t"])
    t2 = max(sgr_cen.data["t"])
    dt = sgr_cen.data["dt"][0]

    # We use the same disk and bulge potentials for all runs, just vary the halo potential
    disk_potential = MiyamotoNagaiPotential(M=1E11*u.M_sun,
                                            a=6.5,
                                            b=0.26)
    bulge_potential = HernquistPotential(M=3.4E10*u.M_sun,
                                         c=0.7)
    potential = disk_potential + bulge_potential + halo_potential

    # Initialize particle simulation with full potential
    simulation = ParticleSimulation(potential=potential)

    for ii in range(sgr_snap.num):
        p = Particle(position=(sgr_snap.data["x"][ii], sgr_snap.data["y"][ii], sgr_snap.data["z"][ii]), # kpc
                     velocity=(sgr_snap.data["vx"][ii], sgr_snap.data["vy"][ii], sgr_snap.data["vz"][ii]), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)

    # The data in SGR_CEN is only printed every 25 steps!
    ts, xs, vs = simulation.run(t1=t2, t2=t1, dt=-dt)
    ts = ts[::-25]
    xs = xs[::-25,:,:]
    vs = vs[::-25,:,:]

    # Define tidal radius, escape velocity for satellite
    msat = 2.5E8 # M_sun
    sgr_orbital_radius = np.sqrt(sgr_cen.data["x"]**2 + sgr_cen.data["y"]**2 + sgr_cen.data["z"]**2)
    m_halo_enclosed = halo_potential.params["v_halo"]**2 * sgr_orbital_radius/bulge_potential.params["_G"]
    mass_enclosed = disk_potential.params["M"] + bulge_potential.params["M"] + m_halo_enclosed

    r_tides = sgr_orbital_radius * (msat / mass_enclosed)**(1./3)
    v_escs = np.sqrt(bulge_potential.params["_G"] * msat / r_tides)

    closest_distances = []
    for ii in range(sgr_snap.num):
        # Distance to satellite center and total velocity
        d = np.sqrt((xs[:,ii,0] - sgr_cen.data["x"])**2 +
                    (xs[:,ii,1] - sgr_cen.data["y"])**2 +
                    (xs[:,ii,2] - sgr_cen.data["z"])**2)
        v = np.sqrt((vs[:,ii,0] - sgr_cen.data["vx"])**2 +
                    (vs[:,ii,1] - sgr_cen.data["vy"])**2 +
                    (vs[:,ii,2] - sgr_cen.data["vz"])**2)

        energy_distances = np.sqrt((d/r_tides)**2 + (v/v_escs)**2)
        closest_distances.append(min(energy_distances))

    return np.array(closest_distances)