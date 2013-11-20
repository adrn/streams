# coding: utf-8
""" """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import pytest

from ... import usys
from ..orbits import *

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


def test_init():

    t = np.arange(0,1000)*u.Myr
    x = np.random.random(size=(10,1000))
    vx = np.random.random(size=(10,1000))
    p = Orbit(t, (x, vx), names=("x","vx"),
              units=[u.kpc, u.km/u.s])
    p = Orbit(t, (x*u.kpc, vx*u.km/u.s), names=("x","vx"))
    assert np.all(p["x"].value == x)
    assert np.allclose(p["vx"].value, vx, rtol=1E-15, atol=1E-15)

    with pytest.raises(TypeError):
        p = Orbit(t.value, (x*u.kpc, vx*u.km/u.s), names=("x","vx"))

def test_getitem():
    t = np.arange(0,1000)*u.Myr
    xyz = np.random.random(size=(3,10,1000))*u.km
    o = Orbit(t, xyz, names=("x","y","z"))
    assert o.nparticles == 10

    o2 = o[4:8]
    assert o2.nparticles == 4
    assert (o2._X[:,0] == o._X[:,4]).all()

def test_repr_X():
    t = np.arange(0,1000)*u.Myr
    x = np.random.random(size=(10,len(t)))
    vx = np.random.random(size=(10,len(t)))
    o = Orbit(t, (x, vx), names=("x","vx"),
                          units=[u.kpc, u.km/u.s])

    assert np.allclose(o._repr_X[0], x)
    assert np.allclose(o._repr_X[1], vx)

def test_plot():
    Ntime = 100
    t = np.arange(0,Ntime)*u.Myr
    x = np.array([[1.],[2.],[3.]]) * np.cos(t.reshape(1,Ntime).value*u.radian)
    y = np.array([[1.],[2.],[3.]]) * np.sin(t.reshape(1,Ntime).value*u.radian)
    vx = -np.array([[1.],[2.],[3.]])*np.sin(t.reshape(1,Ntime).value*u.radian)
    o = Orbit(t, (x, y, vx), names=("x","y","vx"),
                          units=[u.kpc, u.kpc, u.km/u.s])

    fig = o.plot()
    fig.savefig(os.path.join(plot_path, "orbit_2d.png"))

    # now try 6D case
    xx = np.random.random(size=(6,1,Ntime))
    p = Orbit(t, xx, names=("x","y","z","vx","vy","vz"),
              units=[u.kpc, u.kpc, u.kpc, u.km/u.s, u.km/u.s, u.km/u.s])

    fig = p.plot()
    fig.savefig(os.path.join(plot_path, "orbit_6d.png"))
