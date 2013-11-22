# coding: utf-8
""" """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Project
from ..particles import *
from ...observation.error_model import SpitzerGaiaErrorModel
from ... import usys

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# from LM10 -- copied here so I don't have to read in each time...
lm10_X =np.array([[-24.7335,-9.35652,17.7755,-0.266238,-0.013697,-0.00862857],
 [4.04449, -8.58018, 43.6431, -0.147417, -0.00453593, -0.0570219],
 [40.7303, -9.38172, -52.45, 0.10542100000000001, 0.0255395, 0.0676476],
 [31.9827, -0.46618, 3.68191, -0.199363, -0.00786105, 0.16698300000000002],
 [-55.4545, 3.40444, -14.4413, 0.10564799999999999, 0.00946331, -0.0706881],
 [-24.9119, -4.18854, 16.2446, -0.0358232, 0.0333882, -0.20689400000000002],
 [7.63141, 5.4841, 42.3225, -0.238398, -0.02333, -0.0920642],
 [47.9773, -5.17874, -44.0926, 0.0372335, 0.014637, 0.116505],
 [24.7687, 1.91995, -38.9927, 0.13775, 0.0183352, 0.0059355599999999994],
 [35.3565, 15.0336, -55.0006, 0.0649749, -0.0182314,0.124926]]).T

def test_init():

    x = np.random.random(size=100)
    vx = np.random.random(size=100)
    p = Particle((x, vx), names=("x","vx"),
                 units=[u.kpc, u.km/u.s])
    p = Particle((x*u.kpc, vx*u.km/u.s), names=("x","vx"))
    assert np.all(p["x"].value == x)
    assert np.allclose(p["vx"].value, vx, rtol=1E-15, atol=1E-15)

    x_vx = np.random.random(size=(2,100))
    p = Particle(x_vx, names=("x","vx"),
                 units=[u.kpc, u.km/u.s])

    xyz = np.random.random(size=(3,100))*u.km
    p = Particle(xyz, names=("x","y","z"))
    assert p["x"].unit == u.km
    assert p["y"].unit == u.km
    assert p["z"].unit == u.km
    assert np.allclose(p["z"].value, xyz[2], rtol=1E-15, atol=1E-15)

    with pytest.raises(ValueError):
        xyz = np.random.random(size=(3,100))
        p = Particle(xyz, names=("x","y","z"))

    with pytest.raises(ValueError):
        v = np.random.random(size=100)*u.kpc
        p = Particle((v, v.value), names=("x","y"))

def test_getitem():
    xyz = np.random.random(size=(3,100))*u.km
    p = Particle(xyz, names=("x","y","z"))
    assert p.nparticles == 100

    p2 = p[15:30]
    assert p2.nparticles == 15
    assert (p2._X[:,0] == p._X[:,15]).all()

def test_repr_X():
    x = np.random.random(size=100)
    vx = np.random.random(size=100)
    p = Particle((x, vx), names=("x","vx"),
                 units=[u.kpc, u.km/u.s])

    assert np.allclose(p._repr_X[0], x)
    assert np.allclose(p._repr_X[1], vx)

def test_plot():
    x = np.random.random(size=1000)
    vx = np.random.random(size=1000)
    p = Particle((x, vx), names=("x","vx"),
                 units=[u.kpc, u.km/u.s])

    fig = p.plot()
    fig.savefig(os.path.join(plot_path, "particle_2d.png"))

    # now try 6D case
    xx = np.random.random(size=(6,100))
    p = Particle(xx, names=("x","y","z","vx","vy","vz"),
                 units=[u.kpc, u.kpc, u.kpc, u.km/u.s, u.km/u.s, u.km/u.s])

    fig = p.plot()
    fig.savefig(os.path.join(plot_path, "particle_6d.png"))

def test_decompose():
    x = np.random.random(size=1000)
    vx = np.random.random(size=1000)
    p = Particle((x, vx), names=("x","vx"),
                 units=[u.kpc, u.km/u.s])

    p = p.decompose([u.pc, u.radian, u.Gyr, u.M_sun])

    fig = p.plot()
    fig.savefig(os.path.join(plot_path, "particle_2d_decompose.png"))

def test_to_units():
    # now try 6D case
    xx = np.random.random(size=(6,100))
    p = Particle(xx, names=("x","y","z","vx","vy","vz"),
                 units=[u.kpc, u.kpc, u.kpc, u.km/u.s, u.km/u.s, u.km/u.s])

    p = p.to_units(u.pc, u.Gpc, u.km, u.kpc/u.Gyr, u.m/u.ms, u.km/u.s)

    fig = p.plot()
    fig.savefig(os.path.join(plot_path, "particle_6d_to_units.png"))

def test_to_frame():
    # now try 6D case

    p = Particle(lm10_X, names=("x","y","z","vx","vy","vz"),
                units=[u.kpc,u.kpc,u.kpc,u.kpc/u.Myr,u.kpc/u.Myr,u.kpc/u.Myr])

    fig = p.plot(ms=3.)
    fig.savefig(os.path.join(plot_path, "lm10_particle_original.png"))

    p = p.to_frame('heliocentric')
    fig = p.plot(ms=3.)
    fig.savefig(os.path.join(plot_path, "lm10_particle_6d_helio.png"))

    p = p.to_frame('galactocentric')
    fig = p.plot(ms=3.)
    fig.savefig(os.path.join(plot_path, "lm10_particle_6d_galacto.png"))

def test_field_of_streams():
    from ...io import LM10Simulation
    from astropy.coordinates import Galactic

    sgr = LM10Simulation()
    p = sgr.particles(N=10000, expr="(Pcol>-1) & (Pcol<8) & (abs(Lmflag)==1)")
    p2 = p.to_frame("heliocentric")

    icrs = Galactic(p2["l"], p2["b"]).icrs

    g = Galactic(p2["l"], p2["b"], distance=p2["D"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(icrs.ra.degree, icrs.dec.degree, marker='.',
            linestyle='none', alpha=0.25)
    ax.set_xlim(230,110)
    ax.set_ylim(-2,60)
    fig.savefig(os.path.join(plot_path, "field_of_streams_test.png"))

@pytest.mark.parametrize("d_err,", [
    (0.02),
    (0.1),
    (0.5),
])
def test_observe(d_err):
    from ...io import LM10Simulation
    from ...observation.gaia import gaia_spitzer_errors
    sgr = LM10Simulation()
    p = sgr.particles(N=100, expr="(Pcol>-1) & (Pcol<8) & (abs(Lmflag)==1)")
    p = p.to_frame("heliocentric")

    err = gaia_spitzer_errors(p)
    err["D"] = p["D"]*d_err
    o_p = p.observe(err)

    fig = p.plot()
    fig = o_p.plot(fig=fig, color='r')
    fig.savefig(os.path.join(plot_path, "test_observe_hel_{}.png"\
                                        .format(d_err)))

    p = p.to_frame("galactocentric")
    o_p = o_p.to_frame("galactocentric")
    fig = p.plot()
    fig = o_p.plot(fig=fig, color='r')
    fig.savefig(os.path.join(plot_path, "test_observe_gal_{}.png"\
                                        .format(d_err)))