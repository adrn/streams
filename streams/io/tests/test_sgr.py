# coding: utf-8
""" Test different reading data from different mass runs """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
from astropy.constants import G
from astropy.io.misc import fnpickle
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ... import usys
from ..sgr import SgrSimulation
from ...coordinates.frame import heliocentric

plot_path = "plots/tests/io/sgr"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

p_kwargs = dict(marker='.', linestyle='none', color='k', alpha=0.1)
s_kwargs = dict(marker='o', linestyle='none', color='r', alpha=0.75,
                markersize=10)
l_kwargs = dict(marker='^', linestyle='none', color='g', alpha=0.75,
                markersize=10)

class Test10E8(object):

    def setup_class(self):
        self.sgr = SgrSimulation("2.5e8")

        particles = self.sgr.particles(expr="tub==0")
        self.particles = particles.decompose(usys)
        satellite = self.sgr.satellite()
        self.satellite = satellite.decompose(usys)

        # Here are the true parameters from the SCFCEN file
        r0 = np.array([36.82173, 2.926886, -4.172226])*self.sgr._units['length']
        v0 = np.array([4.654394, -0.9905948, 5.080418])*self.sgr._units['length']/self.sgr._units['time']

        self.true_r = np.squeeze(r0.decompose(usys).value)
        self.true_v = np.squeeze(v0.decompose(usys).value)

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

        axes[0,0].plot(self.true_r[0], self.true_r[1], label="Law",
                       **l_kwargs)
        axes[1,0].plot(self.true_r[0], self.true_r[2], **l_kwargs)
        axes[1,1].plot(self.true_r[1], self.true_r[2], **l_kwargs)

        sz = 2
        axes[0,0].set_xlim(self.true_r[0]-sz, self.true_r[0]+sz)
        axes[0,0].set_ylim(self.true_r[1]-sz, self.true_r[1]+sz)

        axes[1,0].set_xlim(self.true_r[0]-sz, self.true_r[0]+sz)
        axes[1,0].set_ylim(self.true_r[2]-sz, self.true_r[2]+sz)

        axes[1,1].set_xlim(self.true_r[1]-sz, self.true_r[1]+sz)
        axes[1,1].set_ylim(self.true_r[2]-sz, self.true_r[2]+sz)

        axes[0,0].legend(fontsize=10)
        fig.subplots_adjust(hspace=0.02,wspace=0.02)
        fig.savefig(os.path.join(plot_path, "sat_ptcl_positions_2.5e8.png"))

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

        axes[0,0].plot(self.true_v[0], self.true_v[1], label="Law", **l_kwargs)
        axes[1,0].plot(self.true_v[0], self.true_v[2], **l_kwargs)
        axes[1,1].plot(self.true_v[1], self.true_v[2], **l_kwargs)

        sz = (50*u.km/u.s).decompose(usys).value
        axes[0,0].set_xlim(self.true_v[0]-sz, self.true_v[0]+sz)
        axes[0,0].set_ylim(self.true_v[1]-sz, self.true_v[1]+sz)

        axes[1,0].set_xlim(self.true_v[0]-sz, self.true_v[0]+sz)
        axes[1,0].set_ylim(self.true_v[2]-sz, self.true_v[2]+sz)

        axes[1,1].set_xlim(self.true_v[1]-sz, self.true_v[1]+sz)
        axes[1,1].set_ylim(self.true_v[2]-sz, self.true_v[2]+sz)

        axes[0,0].legend(fontsize=10)
        fig.subplots_adjust(hspace=0.02,wspace=0.02)
        fig.savefig(os.path.join(plot_path, "sat_ptcl_velocities_2.5e8.png"))

    def test_pickle(self):
        particles = self.sgr.particles(N=25, expr="tub==0")
        fnpickle(particles, os.path.join(plot_path, "test.pickle"))
        p = particles.to_frame(heliocentric)
        fnpickle(p, os.path.join(plot_path, "test2.pickle"))