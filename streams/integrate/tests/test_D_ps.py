# coding: utf-8

""" Test that when back-integrating a bunch of Sgr particles, the D_ps
    distribution looks sensible.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import pytest

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import G

# Project
from ... import usys
from ...io.sgr import SgrSimulation
from ..particle import ParticleIntegrator
from ...potential import LawMajewski2010

# Create logger
logger = logging.getLogger(__name__)

plot_path = "plots/tests/integrate/D_ps"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

np.random.seed(52)
Nparticles = 4
colors = ["#F1A340", "#998EC3", "#67A9CF", "#fff000"]
dt = -1.

true_potential = LawMajewski2010()
wrong_potential = LawMajewski2010(qz=1.8)
simulation = SgrSimulation(mass="2.5e8")
particles = simulation.particles(N=Nparticles,
                                 expr="tub!=0")
satellite = simulation.satellite()

def orbits_from_potential(potential):
    acc = np.zeros((Nparticles+1,3))
    pi = ParticleIntegrator((particles,satellite), potential,
                            args=(Nparticles+1, acc))
    return pi.run(t1=simulation.t1, t2=simulation.t2, dt=dt)


particle_orbit,satellite_orbit = orbits_from_potential(true_potential)

def test_orbits():

    fig,ax = plt.subplots(1,1,figsize=(8,8))
    sgr_fig,sgr_ax = plt.subplots(1,1,figsize=(8,8))

    sgr_orbit = simulation.satellite_orbit
    x = satellite_orbit["x"].to(u.kpc).value
    z = satellite_orbit["z"].to(u.kpc).value
    ax.plot(x, z, color='k', alpha=0.75)

    # plot sgr orbit over the "true" orbit
    true_x = sgr_orbit["x"].to(u.kpc).value
    true_z = sgr_orbit["z"].to(u.kpc).value
    sgr_ax.plot(x, z, color='k', alpha=0.75)
    sgr_ax.plot(true_x,true_z,color='r',alpha=0.5)
    sgr_fig.savefig(os.path.join(plot_path, "sgr_orbit.png"))

    for ii in range(Nparticles):
        x = particle_orbit["x"][:,ii].to(u.kpc).value
        z = particle_orbit["z"][:,ii].to(u.kpc).value
        ax.plot(x, z, color=colors[ii], alpha=0.75)

    fig.savefig(os.path.join(plot_path, "orbits.png"))

def test_dps():
    for jj,potential in enumerate([true_potential,wrong_potential]):
        particle_orbit,satellite_orbit = orbits_from_potential(potential)
        sat_var = np.zeros((len(particle_orbit.t),6))
        sat_var[:,:3] = potential._tidal_radius(satellite.m,
                                                satellite_orbit._X[...,:3])*1.26
        sat_var[:,3:] += satellite.v_disp
        cov = (sat_var**2)[:,np.newaxis]

        D_ps = np.sqrt(np.sum((particle_orbit._X - satellite_orbit._X)**2 / cov, \
                              axis=-1))

        fig,ax = plt.subplots(1,1,figsize=(12,6))
        for ii in range(Nparticles):
            ax.plot(particle_orbit.t.value, D_ps[:,ii],
                    color=colors[ii], alpha=0.8)
            ax.axvline(particles.tub[ii], color=colors[ii], alpha=0.8)

        ax.set_ylim(0,5)
        ax.axhline(1.4)
        fig.savefig(os.path.join(plot_path, "d_ps_{}.png".format(jj)))
