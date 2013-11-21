# coding: utf-8
"""
    Make sure the satellite starting position coincides with the particles
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ... import usys
from ..lm10 import LM10Simulation

plot_path = "plots/tests/io/lm10"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

lm10 = LM10Simulation()
particles = lm10.particles(expr="Pcol==-1")
particles = particles.decompose(usys)
satellite = lm10.satellite()
satellite = satellite.decompose(usys)

# Here are the true parameters from the last block in R601LOG
GG = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
X = (GG / 0.85**3 * 6.4E8)**-0.5
length_unit = u.Unit("0.85 kpc")
mass_unit = u.Unit("6.4E8 M_sun")
time_unit = u.Unit("{:08f} Myr".format(X))
r0 = np.array([[2.3279727753E+01,2.8190329987,-6.8798148785]])*length_unit
v0 = np.array([[3.9481694047,-6.1942673069E-01,3.4555581435]])*length_unit/time_unit

law_r = np.squeeze(r0.decompose(usys).value)
law_v = np.squeeze(v0.decompose(usys).value)

def test_position():
    p_kwargs = dict(marker='.', linestyle='none', color='k', alpha=0.1)
    s_kwargs = dict(marker='o', linestyle='none', color='r', alpha=0.75)
    l_kwargs = dict(marker='o', linestyle='none', color='g', alpha=0.75)

    #fig,axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,10))
    fig,axes = plt.subplots(2, 2, figsize=(10,10))
    axes[0,1].set_visible(False)

    axes[0,0].plot(particles["x"].value,
                   particles["y"].value,
                   label="all particles", **p_kwargs)
    axes[1,0].plot(particles["x"].value,
                   particles["z"].value,
                   **p_kwargs)
    axes[1,1].plot(particles["y"].value,
                   particles["z"].value,
                   **p_kwargs)

    axes[0,0].plot(satellite["x"].value,
                   satellite["y"].value,
                   label="Satellite", **s_kwargs)
    axes[1,0].plot(satellite["x"].value,
                   satellite["z"].value,
                   **s_kwargs)
    axes[1,1].plot(satellite["y"].value,
                   satellite["z"].value,
                   **s_kwargs)

    axes[0,0].plot(law_r[0], law_r[1], label="Law", **l_kwargs)
    axes[1,0].plot(law_r[0], law_r[2], **l_kwargs)
    axes[1,1].plot(law_r[1], law_r[2], **l_kwargs)

    sz = 5
    axes[0,0].set_xlim(law_r[0]-5, law_r[0]+5)
    axes[0,0].set_ylim(law_r[1]-5, law_r[1]+5)

    axes[1,0].set_xlim(law_r[0]-5, law_r[0]+5)
    axes[1,0].set_ylim(law_r[2]-5, law_r[2]+5)

    axes[1,1].set_xlim(law_r[1]-5, law_r[1]+5)
    axes[1,1].set_ylim(law_r[2]-5, law_r[2]+5)

    axes[0,0].legend(fontsize=10)
    fig.subplots_adjust(hspace=0.02,wspace=0.02)
    fig.savefig(os.path.join(plot_path, "sat_ptcl_positions.png"))

def test_velocity():

    fig,axes = plt.subplots(2, 2)

    axes[0,1].set_visible(False)

    axes[0,0].plot(particles._v[...,0], particles._v[...,1], marker='.',
                   linestyle='none', color='k', alpha=0.1)
    axes[1,0].plot(particles._v[...,0], particles._v[...,2], marker='.',
                   linestyle='none', color='k', alpha=0.1)
    axes[1,1].plot(particles._v[...,1], particles._v[...,2], marker='.',
                   linestyle='none', color='k', alpha=0.1)

    axes[0,0].plot(satellite._v[...,0], satellite._v[...,1], marker='o',
                   linestyle='none', color='r')
    axes[1,0].plot(satellite._v[...,0], satellite._v[...,2], marker='o',
                   linestyle='none', color='r')
    axes[1,1].plot(satellite._v[...,1], satellite._v[...,2], marker='o',
                   linestyle='none', color='r')

    mean_v = np.mean(particles._v, axis=0)
    axes[0,0].plot(mean_v[0], mean_v[1], marker='o',
                   linestyle='none', color='b', alpha=0.4)
    axes[1,0].plot(mean_v[0], mean_v[2], marker='o',
                   linestyle='none', color='b', alpha=0.4)
    axes[1,1].plot(mean_v[1], mean_v[2], marker='o',
                   linestyle='none', color='b', alpha=0.4)

    axes[0,0].plot(law_v[0], law_v[1], marker='o',
                   linestyle='none', color='g', alpha=0.4)
    axes[1,0].plot(law_v[0], law_v[2], marker='o',
                   linestyle='none', color='g', alpha=0.4)
    axes[1,1].plot(law_v[1], law_v[2], marker='o',
                   linestyle='none', color='g', alpha=0.4)

    axes[0,0].set_xlim(mean_v[0]-0.02,mean_v[0]+0.02)
    axes[1,0].set_xlim(mean_v[0]-0.02,mean_v[0]+0.02)
    axes[1,1].set_xlim(mean_v[1]-0.02,mean_v[1]+0.02)

    axes[0,0].set_ylim(mean_v[1]-0.02,mean_v[1]+0.02)
    axes[1,0].set_ylim(mean_v[2]-0.02,mean_v[2]+0.02)
    axes[1,1].set_ylim(mean_v[2]-0.02,mean_v[2]+0.02)

    plt.savefig(os.path.join(plot_path, "sat_part_velocities.png"))