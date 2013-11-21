# coding: utf-8
""" Test different reading data from an orphan-like progenitor """

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
from ..orphan import OrphanSimulation

plot_path = "plots/tests/io/orp"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

p_kwargs = dict(marker='.', linestyle='none', color='k', alpha=0.1)
s_kwargs = dict(marker='o', linestyle='none', color='r', alpha=0.75,
                markersize=10)
l_kwargs = dict(marker='^', linestyle='none', color='g', alpha=0.75,
                markersize=10)

class TestOrphan(object):

    def setup_class(self):
        self.sgr = OrphanSimulation()

        particles = self.sgr.particles(expr="tub==0")
        self.particles = particles.decompose(usys)
        satellite = self.sgr.satellite()
        self.satellite = satellite.decompose(usys)

    def test_position(self):
        fig,axes = plt.subplots(2, 2, figsize=(10,10))
        axes[0,1].set_visible(False)

        axes[0,0].plot(self.particles["x"].value,
                       self.particles["y"].value,
                       label="all particles", **p_kwargs)
        axes[1,0].plot(self.particles["x"].value,
                       self.particles["z"].value,
                       **p_kwargs)
        axes[1,1].plot(self.particles["y"].value,
                       self.particles["z"].value,
                       **p_kwargs)

        axes[0,0].plot(self.satellite["x"].value,
                       self.satellite["y"].value,
                       label="Satellite", **s_kwargs)
        axes[1,0].plot(self.satellite["x"].value,
                       self.satellite["z"].value,
                       **s_kwargs)
        axes[1,1].plot(self.satellite["y"].value,
                       self.satellite["z"].value,
                       **s_kwargs)

        axes[0,0].legend(fontsize=10)
        fig.subplots_adjust(hspace=0.02,wspace=0.02)
        fig.savefig(os.path.join(plot_path, "sat_ptcl_positions.png"))

    def test_velocity(self):
        fig,axes = plt.subplots(2, 2, figsize=(10,10))
        axes[0,1].set_visible(False)

        axes[0,0].plot(self.particles["vx"].value,
                       self.particles["vy"].value,
                       label="all particles", **p_kwargs)
        axes[1,0].plot(self.particles["vx"].value,
                       self.particles["vz"].value,
                       **p_kwargs)
        axes[1,1].plot(self.particles["vy"].value,
                       self.particles["vz"].value,
                       **p_kwargs)

        axes[0,0].plot(self.satellite["vx"].value,
                       self.satellite["vy"].value,
                       label="Satellite", **s_kwargs)
        axes[1,0].plot(self.satellite["vx"].value,
                       self.satellite["vz"].value,
                       **s_kwargs)
        axes[1,1].plot(self.satellite["vy"].value,
                       self.satellite["vz"].value,
                       **s_kwargs)

        axes[0,0].legend(fontsize=10)
        fig.subplots_adjust(hspace=0.02,wspace=0.02)
        fig.savefig(os.path.join(plot_path, "sat_ptcl_velocities.png"))