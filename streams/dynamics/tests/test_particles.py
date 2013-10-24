# coding: utf-8
""" """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import astropy.units as u
import numpy as np
import pytest

# Project
from ..particles import *
from ...observation.gaia import RRLyraeErrorModel
from ... import usys

plot_path = "plots/tests/dynamics"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

units = (u.kpc, u.Myr, u.M_sun)

def test_Particle_init():

    # Init with individual arrays of ndim=1
    r = np.random.random(3)*u.kpc
    v = np.random.random(3)*u.km/u.s
    m = np.random.random()*u.M_sun

    with pytest.raises(ValueError):
        pc = Particle(r=r, v=v, m=m, units=units)

    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.kpc/u.Myr
    m = np.random.random(10)*u.M_sun

    pc = Particle(r=r, v=v, m=m, units=units)

    assert np.all(pc.r.value == r.value)
    assert np.all(pc.v.value == v.value)
    assert np.all(pc.m.value == m.value)

    assert np.all(pc._r == r.value)
    assert np.all(pc._v == v.value)
    assert np.all(pc._m == m.value)

def test_acceleration():
    r = np.array([[1.,0.],
                  [0, 1.],
                  [-1., 0.],
                  [0., -1.]])*u.kpc
    v = np.zeros_like(r.value)*u.km/u.s
    m = np.random.random()*u.M_sun

    pc = Particle(r=r, v=v, m=m, units=units)

    pc.acceleration_at(np.array([0.,0.])*u.kpc)

    a = pc.acceleration_at(np.array([[0.5,0.5], [0.0,0.0], [-0.5, -0.5]])*u.kpc)

def test_merge():
    # test merging two particle collections

    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun

    pc1 = Particle(r=r, v=v, m=m, units=units)

    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.km/u.s
    m = np.random.random(10)*u.M_sun

    pc2 = Particle(r=r, v=v, m=m, units=units)

    pc_merged = pc1.merge(pc2)

    assert pc_merged._r.shape == (20,3)

def test_to():
    r = np.random.random(size=(10,3))*u.kpc
    v = np.random.random(size=(10,3))*u.kpc/u.Myr
    m = np.random.random(10)*u.M_sun

    pc = Particle(r=r, v=v, m=m, units=units)

    usys2 = (u.km, u.s, u.kg)
    pc2 = pc.to(usys2)

    assert np.allclose(pc2._r, r.to(u.km).value)
    assert np.allclose(pc2._v, v.to(u.km/u.s).value)
    assert np.allclose(pc2._m, m.to(u.kg).value)

def test_getitem():
    r = np.random.random(size=(100,3))*u.kpc
    v = np.random.random(size=(100,3))*u.kpc/u.Myr
    m = np.random.random(100)*u.M_sun

    pc = Particle(r=r, v=v, m=m, units=units)
    pc2 = pc[15:30]

    assert pc2.nparticles == 15
    assert (pc2._r[0] == pc._r[15]).all()

def test_plot():
    r = np.random.random(size=(100,3))*u.kpc
    v = np.random.random(size=(100,3))*u.kpc/u.Myr
    m = np.random.random(100)*u.M_sun
    p = Particle(r=r, v=v, m=m, units=units)

    fig = p.plot_r()
    fig.savefig(os.path.join(plot_path, "particle_r.png"))

    fig = p.plot_v()
    fig.savefig(os.path.join(plot_path, "particle_v.png"))

def test_observe():

    # from LM10 -- copied here so I don't have to read in each time...
    r = np.array([[-24.7335, -9.35652, 17.7755], [4.04449, -8.58018, 43.6431], [40.7303, -9.38172, -52.45], [31.9827, -0.46618, 3.68191], [-55.4545, 3.40444, -14.4413], [-24.9119, -4.18854, 16.2446], [7.63141, 5.4841, 42.3225], [47.9773, -5.17874, -44.0926], [24.7687, 1.91995, -38.9927], [35.3565, 15.0336, -55.0006]]) * u.kpc
    v = np.array([[-0.26623800000000003, -0.013696999999999999, -0.00862857], [-0.147417, -0.00453593, -0.0570219], [0.10542100000000001, 0.0255395, 0.0676476], [-0.199363, -0.00786105, 0.16698300000000002], [0.10564799999999999, 0.00946331, -0.0706881], [-0.0358232, 0.0333882, -0.20689400000000002], [-0.238398, -0.02333, -0.0920642], [0.0372335, 0.014637, 0.116505], [0.13775, 0.0183352, 0.0059355599999999994], [0.0649749, -0.018231400000000002, 0.12492600000000001]]) * u.kpc/u.Myr

    p = Particle(r=r, v=v, m=1*u.M_sun)

    for factor in [0.01, 0.1, 1., 10.]:
        error_model = RRLyraeErrorModel(units=usys, factor=factor)
        new_p = p.observe(error_model)

        fig = p.plot_r()
        new_p.plot_r(fig=fig, color="r")
        fig.savefig(os.path.join(plot_path,
                    "particle_observed_r_fac{0}.png".format(factor)))

        fig = p.plot_v()
        new_p.plot_v(fig=fig, color="r")
        fig.savefig(os.path.join(plot_path,
                    "particle_observed_v_fac{0}.png".format(factor)))

def test_time_observe():
    r = np.random.random(size=(100,3))*u.kpc
    v = np.random.random(size=(100,3))*u.kpc/u.Myr
    m = np.random.random(100)*u.M_sun
    p = Particle(r=r, v=v, m=m, units=usys)

    error_model = RRLyraeErrorModel(units=usys, factor=1.)
    a = time.time()
    for ii in range(10):
        new_p = p.observe(error_model)
    print((time.time()-a)/10.)