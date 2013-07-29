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

from ..lm10 import particles_today, satellite_today, time, satellite_orbit

plot_path = "plots/tests/io/lm10"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

satellite = satellite_today()
particles = particles_today(N=0, expr="(Pcol==-1)")

# from email
law_r = [19.0149, 2.64883, -6.8686]
law_v = [230.2018, -35.18828, 194.7525]

def test_position():
    
    fig,axes = plt.subplots(2, 2)
    
    axes[0,1].set_visible(False)
    
    axes[0,0].plot(particles._r[...,0], particles._r[...,1], marker='.', 
                   linestyle='none', color='k', alpha=0.1)
    axes[1,0].plot(particles._r[...,0], particles._r[...,2], marker='.', 
                   linestyle='none', color='k', alpha=0.1)
    axes[1,1].plot(particles._r[...,1], particles._r[...,2], marker='.',
                   linestyle='none', color='k', alpha=0.1)
    
    axes[0,0].plot(satellite._r[...,0], satellite._r[...,1], marker='o', 
                   linestyle='none', color='r')
    axes[1,0].plot(satellite._r[...,0], satellite._r[...,2], marker='o', 
                   linestyle='none', color='r')
    axes[1,1].plot(satellite._r[...,1], satellite._r[...,2], marker='o', 
                   linestyle='none', color='r')
    
    mean_r = np.mean(particles._r, axis=0)
    axes[0,0].plot(mean_r[0], mean_r[1], marker='o', 
                   linestyle='none', color='b', alpha=0.4)
    axes[1,0].plot(mean_r[0], mean_r[2], marker='o', 
                   linestyle='none', color='b', alpha=0.4)
    axes[1,1].plot(mean_r[1], mean_r[2], marker='o', 
                   linestyle='none', color='b', alpha=0.4)
    
    axes[0,0].plot(law_r[0], law_r[1], marker='o', 
                   linestyle='none', color='g', alpha=0.4)
    axes[1,0].plot(law_r[0], law_r[2], marker='o', 
                   linestyle='none', color='g', alpha=0.4)
    axes[1,1].plot(law_r[1], law_r[2], marker='o', 
                   linestyle='none', color='g', alpha=0.4)
    
    axes[0,0].set_xlim(mean_r[0]-1.,mean_r[0]+1.)
    axes[1,0].set_xlim(mean_r[0]-1.,mean_r[0]+1.)
    axes[1,1].set_xlim(mean_r[1]-1.,mean_r[1]+1.)
    
    axes[0,0].set_ylim(mean_r[1]-1.,mean_r[1]+1.)
    axes[1,0].set_ylim(mean_r[2]-1.,mean_r[2]+1.)
    axes[1,1].set_ylim(mean_r[2]-1.,mean_r[2]+1.)
    
    plt.savefig(os.path.join(plot_path, "sat_part_positions.png"))

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