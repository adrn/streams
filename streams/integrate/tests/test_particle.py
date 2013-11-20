# coding: utf-8

""" Test the particle integrator """

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
from ...dynamics import Particle
from ..particle import ParticleIntegrator
from ...potential import AxisymmetricNFWPotential, PointMassPotential

# Create logger
logger = logging.getLogger(__name__)

plot_path = "plots/tests/integrate/ParticleIntegrator"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_init():
    X = np.random.random(size=(6,100))
    p = Particle(X,
                 names=("x","y","z","vx","vy","vz"),
                 units=(u.kpc,u.kpc,u.kpc,u.km/u.s,u.km/u.s,u.km/u.s))
    potential = PointMassPotential(units=usys, m=1E12*u.M_sun)
    pi = ParticleIntegrator(p,potential)

    X = np.random.random(size=(6,100))
    X[3:] = X[3:]*50
    p2 = Particle(X,
                  names=("x","y","z","vx","vy","vz"),
                  units=(u.kpc,u.kpc,u.kpc,u.km/u.s,u.km/u.s,u.km/u.s))
    pi = ParticleIntegrator((p,p2),potential)

def test_simple_run():
    # test with a disk around a point mass?
    N = 100
    m = 1E12*u.M_sun

    r = np.sqrt(np.random.uniform(0.5, 8, size=N))
    phi = np.random.uniform(0., 2*np.pi, size=N)
    _G = G.decompose(usys).value
    V = np.sqrt(_G*m / r)

    X = np.zeros((6,N))
    X[0] = r*np.cos(phi)
    X[1] = r*np.sin(phi)
    X[2] = np.random.uniform(-0.2,0.2,size=N)
    X[3] = -V*np.sin(phi)
    X[4] = V*np.cos(phi)

    p = Particle(X,
                names=("x","y","z","vx","vy","vz"),
                units=(u.kpc,u.kpc,u.kpc,u.kpc/u.Myr,u.kpc/u.Myr,u.kpc/u.Myr))
    potential = PointMassPotential(units=usys, m=m)
    pi = ParticleIntegrator(p,potential)
    orbit = pi.run(t1=0., t2=500., dt=0.1)[0]

    # initial conditions
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot(orbit["x"][0].value, orbit["y"][0].value,
             marker='.', alpha=0.5, linestyle='none')
    fig.savefig(os.path.join(plot_path, "simple_initial.png"))

def test_run_pretty():
    # test with a disk around a point mass?
    N = 100
    m = 1E12*u.M_sun

    r = np.sqrt(np.random.uniform(0.5, 8, size=N))
    phi = np.random.uniform(0., 2*np.pi, size=N)
    _G = G.decompose(usys).value
    V = np.sqrt(_G*m / r)

    X = np.zeros((6,N))
    X[0] = r*np.cos(phi)
    X[1] = r*np.sin(phi)
    X[2] = np.random.uniform(-0.2,0.2,size=N)
    X[3] = -V*np.sin(phi)
    X[4] = V*np.cos(phi)

    p = Particle(X,
                names=("x","y","z","vx","vy","vz"),
                units=(u.kpc,u.kpc,u.kpc,u.kpc/u.Myr,u.kpc/u.Myr,u.kpc/u.Myr))
    potential = AxisymmetricNFWPotential(units=usys,
                                         log_m=np.log(m.value),
                                         qz=0.8,
                                         Rs=1*u.kpc)
    pi = ParticleIntegrator(p,potential)

    orbit = pi.run(t1=0., t2=500., dt=0.1)[0]

    # initial conditions
    plt.figure(figsize=(6,6))
    plt.plot(X[0], X[1], marker='.', alpha=0.5, linestyle='none')
    plt.savefig(os.path.join(plot_path, "initial.png"))

    # orbit of one star
    fig,axes = plt.subplots(3,1,sharex=True)
    for ii,n in zip(range(3),orbit.names[:3]):
        axes[ii].plot(orbit.t.value, orbit[n][0].value)
    fig.savefig(os.path.join(plot_path, "one_xyz.png"))

    apos = np.max(np.sqrt(np.sum(orbit._X[:3]**2, axis=0)), axis=-1)
    fig, ax = plt.subplots(1,1,figsize=(8.5,11))
    ax.set_color_cycle(['#9E0142', '#D53E4F', '#F46D43', '#FDAE61', \
                        '#FEE08B', '#E6F598', '#ABDDA4', '#66C2A5', \
                        '#3288BD', '#5E4FA2'])
    for ii in range(N):
        ax.plot(orbit["x"][ii].value, orbit["y"][ii].value,
                alpha=(apos[ii]/max(apos))**1.8*0.99+0.01,
                linestyle='-', lw=2., marker=None)

    ax.set_axis_bgcolor("#333333")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(-20,20)
    ax.set_ylim(-25.88,25.88)
    fig.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    fig.savefig(os.path.join(plot_path, "all_orbit.pdf"), facecolor='#333333')