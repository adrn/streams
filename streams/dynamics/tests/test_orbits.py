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
    x = np.random.random(size=100)
    vx = np.random.random(size=100)
    p = Orbit(t, (x, vx), names=("x","vx"),
              units=[u.kpc, u.km/u.s])
    p = Orbit(t, (x*u.kpc, vx*u.km/u.s), names=("x","vx"))
    assert np.all(p["x"].value == x)
    assert np.allclose(p["vx"].value, vx, rtol=1E-15, atol=1E-15)

    with pytest.raises(TypeError):
        p = Orbit(t.value, (x*u.kpc, vx*u.km/u.s), names=("x","vx"))

def test_getitem():
    xyz = np.random.random(size=(3,100))*u.km
    p = Orbit(xyz, names=("x","y","z"))
    assert len(p) == 100

    p2 = p[15:30]
    assert len(p2) == 15
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

def test_orbitcollection_init():
    t = np.linspace(0., 10., 100)*u.yr
    r = np.random.random((100,10,3))*u.kpc
    v = np.random.random((100,10,3))*u.kpc/u.yr
    m = np.random.random(10)*u.M_sun

    oc = Orbit(t=t, r=r, v=v, m=m)

    # Should fail -- different length units, no usys:
    t = np.linspace(0., 10., 100)*u.yr
    r = np.random.random((100,10,3))*u.kpc
    v = np.random.random((100,10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun
    with pytest.raises(ValueError):
        oc = Orbit(t=t, r=r, v=v, m=m)

    # should pass bc we give a usys
    oc = Orbit(t=t, r=r, v=v, m=m, units=usys)

def test_to():
    t = np.arange(0., 100, 0.1)*u.Myr
    r = np.random.random(size=(len(t),10,3))*u.kpc
    v = np.random.random(size=(len(t),10,3))*u.kpc/u.Myr
    m = np.random.random(10)*u.M_sun

    pc = Orbit(t=t, r=r, v=v, m=m, units=usys)

    usys2 = (u.km, u.s, u.kg)
    pc2 = pc.to(usys2)

    assert np.allclose(pc2._t, t.to(u.s).value)
    assert np.allclose(pc2._r, r.to(u.km).value)
    assert np.allclose(pc2._v, v.to(u.km/u.s).value)
    assert np.allclose(pc2._m, m.to(u.kg).value)

def test_slice():
    t = np.arange(0., 100, 0.1)*u.Myr
    r = np.random.random(size=(len(t),10,3))*u.kpc
    v = np.random.random(size=(len(t),10,3))*u.kpc/u.Myr
    m = np.random.random(10)*u.M_sun

    pc = Orbit(t=t, r=r, v=v, m=m, units=usys)

    assert isinstance(pc[0], Particle)
    assert isinstance(pc[0:15], Orbit)

def test_plot():
    t = np.arange(0., 100, 0.1)*u.Myr
    r = np.random.random(size=(len(t),10,3))*u.kpc
    v = np.random.random(size=(len(t),10,3))*u.kpc/u.Myr
    m = np.random.random(10)*u.M_sun

    oc = Orbit(t=t, r=r, v=v, m=m, units=usys)

    fig = oc.plot_r()
    fig.savefig(os.path.join(plot_path, "orbit_r.png"))

    fig = oc.plot_v()
    fig.savefig(os.path.join(plot_path, "orbit_v.png"))

