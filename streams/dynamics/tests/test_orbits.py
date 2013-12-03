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
from ...coordinates.frame import ReferenceFrame, galactocentric, heliocentric

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

f1 = ReferenceFrame(name="test",
                    coord_names=("x","vx"),
                    units=[u.kpc, u.km/u.s])

f2 = ReferenceFrame(name="cartesian",
                    coord_names=("x","y","z"),
                    units=[u.kpc, u.kpc, u.kpc])

f3 = ReferenceFrame(name="test",
                    coord_names=("x","y","vx"),
                    units=[u.kpc, u.kpc, u.km/u.s])

def test_init():

    t = np.arange(0,1000)*u.Myr
    x = np.random.random(size=(1000,10))
    vx = np.random.random(size=(1000,10))
    p = Orbit(t, (x, vx), frame=f1, units=f1.units)
    p = Orbit(t, (x*u.kpc, vx*u.km/u.s), frame=f1)

    assert np.all(p["x"].value == x)
    assert np.allclose(p["vx"].value, vx, rtol=1E-15, atol=1E-15)

    with pytest.raises(TypeError):
        p = Orbit(t.value, (x*u.kpc, vx*u.km/u.s), frame=f1)

def test_getitem():
    t = np.arange(0,1000)*u.Myr
    xyz = np.random.random(size=(3,1000,10))*u.km
    o = Orbit(t, xyz, frame=f2)
    assert o.nparticles == 10

    with pytest.raises(ValueError):
        o2 = o[4:8]

def test_repr_X():
    t = np.arange(0,1000)*u.Myr
    x = np.random.random(size=(10,len(t)))
    vx = np.random.random(size=(10,len(t)))
    o = Orbit(t, (x, vx), frame=f1, units=f1.units)

    assert np.allclose(o._repr_X[...,0], x.T)
    assert np.allclose(o._repr_X[...,1], vx.T)

def test_plot():
    Ntime = 100
    t = np.arange(0,Ntime)*u.Myr
    _w = np.array([1.,2.,3.]).reshape(1,3)
    x = _w * np.cos(t.reshape(Ntime,1).value*u.radian)
    y = _w * np.sin(t.reshape(Ntime,1).value*u.radian)
    vx = -_w*np.sin(t.reshape(Ntime,1).value*u.radian)

    o = Orbit(t, (x, y, vx), frame=f3, units=f3.units)

    fig = o.plot()
    fig.savefig(os.path.join(plot_path, "orbit_2d.png"))

    # now try 6D case
    f = ReferenceFrame(name="cartesian",
                    coord_names=("x","y","z","vx","vy","vz"),
                    units=[u.kpc, u.kpc, u.kpc, u.km/u.s, u.km/u.s, u.km/u.s])
    xx = np.random.random(size=(6,Ntime,1))
    p = Orbit(t, xx, frame=f, units=f.units)

    fig = p.plot()
    fig.savefig(os.path.join(plot_path, "orbit_6d.png"))
