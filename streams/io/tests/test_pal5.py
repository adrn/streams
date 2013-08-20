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

from ..pal5 import particle_table, particles_today, satellite_today, time

plot_path = "plots/tests/io/pal5"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

data = particle_table(N=0)

def test_position():
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
    fig.savefig(os.path.join(plot_path, "pos.png"))

def test_velocity():
    fig,axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,12) )
    axes[0,1].set_visible(False)
    axes[0,0].plot(data['vx'], data['vy'], marker='.', 
                   linestyle='none', color='k', alpha=0.1)
    axes[1,0].plot(data['vx'], data['vz'], marker='.', 
                   linestyle='none', color='k', alpha=0.1)
    axes[1,1].plot(data['vy'], data['vz'], marker='.',
                   linestyle='none', color='k', alpha=0.1)

    fig.savefig(os.path.join(plot_path, "vel.png"))
    
def test_particles_today():
    particles = particles_today(N=0)
    fig = particles.plot_r()
    fig.savefig(os.path.join(plot_path, "particles_r.png"))
