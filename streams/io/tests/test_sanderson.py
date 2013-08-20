# coding: utf-8
"""
    Make sure the satellite starting position coincides with the particles
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..sanderson import particle_table, particles_today, satellite_today, \
                            time, satellite_orbit

plot_path = "plots/tests/io/gaia_challenge"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

ii = 86

def test_position():
    data = particle_table(N=0, satellite_id=ii)
    fig,axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,12) )
    axes[0,1].set_visible(False)
    axes[0,0].plot(data['x'], data['y'], marker='.', 
                   linestyle='none', color='k', alpha=0.1)
    axes[1,0].plot(data['x'], data['z'], marker='.', 
                   linestyle='none', color='k', alpha=0.1)
    axes[1,1].plot(data['y'], data['z'], marker='.',
                   linestyle='none', color='k', alpha=0.1)
    axes[0,0].set_xlim(-60,60)
    axes[0,0].set_ylim(-60,60)
    fig.savefig(os.path.join(plot_path, "sat{0}_pos.png".format(ii)))

def test_velocity():
    data = particle_table(N=0, satellite_id=ii)
    fig,axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,12) )
    axes[0,1].set_visible(False)
    axes[0,0].plot(data['vx'], data['vy'], marker='.', 
                   linestyle='none', color='k', alpha=0.1)
    axes[1,0].plot(data['vx'], data['vz'], marker='.', 
                   linestyle='none', color='k', alpha=0.1)
    axes[1,1].plot(data['vy'], data['vz'], marker='.',
                   linestyle='none', color='k', alpha=0.1)

    fig.savefig(os.path.join(plot_path, "sat{0}_vel.png".format(ii)))
    
def test_particles_today():
    particles = particles_today(N=0, satellite_id=ii)
    
    fig = particles.plot_r()
    fig.savefig(os.path.join(plot_path, "sat{0}_particles.png".format(ii)))
