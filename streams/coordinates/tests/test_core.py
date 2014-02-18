# coding: utf-8
"""
    Test conversions in core.py
"""

from __future__ import absolute_import, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np

import astropy.coordinates as coord
import astropy.units as u

from ... import usys
from ..core import *

this_path = os.path.split(__file__)[0]
data = np.genfromtxt(os.path.join(this_path, "idl_vgsr_vhel.txt"),
                     names=True, skiprows=2)

def test_gsr_to_hel():
    for row in data:
        l = coord.Angle(row["lon"] * u.degree)
        b = coord.Angle(row["lat"] * u.degree)
        vgsr = row["vgsr"] * u.km/u.s
        vlsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s

        vhel = vgsr_to_vhel(l, b, vgsr,
                             vlsr=vlsr,
                             vcirc=row["vcirc"]*u.km/u.s)

        np.testing.assert_almost_equal(vhel.value, row['vhelio'], decimal=4)

def test_gsr_to_hel_lon():
    l1 = coord.Angle(190.*u.deg)
    l2 = coord.Angle(-170.*u.deg)
    b = coord.Angle(30.*u.deg)
    vgsr = -110.*u.km/u.s

    vhel1 = vgsr_to_vhel(l1,b,vgsr)
    vhel2 = vgsr_to_vhel(l2,b,vgsr)

    np.testing.assert_almost_equal(vhel1.value, vhel2.value, decimal=9)

def test_hel_to_gsr():
    for row in data:
        l = coord.Angle(row["lon"] * u.degree)
        b = coord.Angle(row["lat"] * u.degree)
        vhel = row["vhelio"] * u.km/u.s
        vlsr = [row["vx"],row["vy"],row["vz"]]*u.km/u.s

        vgsr = vhel_to_vgsr(l, b, vhel,
                             vlsr=vlsr,
                             vcirc=row["vcirc"]*u.km/u.s)

        np.testing.assert_almost_equal(vgsr.value, row['vgsr'], decimal=4)

def test_roundtrip():
    np.random.seed(43)
    N = 100
    x1 = np.random.uniform(-50.,50.,size=N)*u.kpc
    y1 = np.random.uniform(-50.,50.,size=N)*u.kpc
    z1 = np.random.uniform(-50.,50.,size=N)*u.kpc

    vx1 = np.random.uniform(-100.,100.,size=N)*u.km/u.s
    vy1 = np.random.uniform(-100.,100.,size=N)*u.km/u.s
    vz1 = np.random.uniform(-100.,100.,size=N)*u.km/u.s

    l,b,d,mul,mub,vr = gc_to_hel(x1,y1,z1,vx1,vy1,vz1)
    x2,y2,z2,vx2,vy2,vz2 = hel_to_gc(l,b,d,mul,mub,vr)

    np.all(np.round((x1-x2)/x1*100.,2) == 0)
    np.all(np.round((y1-y2)/y1*100.,2) == 0)
    np.all(np.round((z1-z2)/z1*100.,2) == 0)
    np.all(np.round((vx1-vx2)/vx1*100.,2) == 0)
    np.all(np.round((vy1-vy2)/vy1*100.,2) == 0)
    np.all(np.round((vz1-vz2)/vz1*100.,2) == 0)

def test_gal_to_hel_call():

    r = np.random.uniform(-10,10,size=(3,1000))*u.kpc
    v = np.random.uniform(-100,100,size=(3,1000))*u.km/u.s

    gal_xyz_to_hel_lbd(r)
    gal_xyz_to_hel_lbd(r, v)

def test_hel_to_gal():

    # l = 0
    r,v = hel_lbd_to_gal_xyz((0*u.deg, 0*u.deg, 2*u.kpc),
                             (0*u.mas/u.yr, 0*u.mas/u.yr, 20*u.km/u.s),
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r, [-6,0,0]*u.kpc)
    np.testing.assert_almost_equal(v, [20,200,0]*u.km/u.s)

    # l = 90
    r,v = hel_lbd_to_gal_xyz((90*u.deg, 0*u.deg, 2*u.kpc),
                             (0*u.mas/u.yr, 0*u.mas/u.yr, 20*u.km/u.s),
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r, [-8,2,0]*u.kpc)
    np.testing.assert_almost_equal(v, [0,220,0]*u.km/u.s)

    # l = 180
    r,v = hel_lbd_to_gal_xyz((180*u.deg, 0*u.deg, 2*u.kpc),
                             (0*u.mas/u.yr, 0*u.mas/u.yr, 20*u.km/u.s),
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r, [-10,0,0]*u.kpc)
    np.testing.assert_almost_equal(v, [-20,200,0]*u.km/u.s)

    # l = 270
    r,v = hel_lbd_to_gal_xyz((270*u.deg, 0*u.deg, 2*u.kpc),
                             (0*u.mas/u.yr, 0*u.mas/u.yr, 20*u.km/u.s),
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r, [-8,-2,0]*u.kpc)
    np.testing.assert_almost_equal(v, [0,180,0]*u.km/u.s)

    print(r,v)

def test_gal_to_hel():

    # l = 0
    r,v = gal_xyz_to_hel_lbd([-6,0,0]*u.kpc,
                             [20,200,0]*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r[0], 0*u.deg)
    np.testing.assert_almost_equal(r[1], 0*u.deg)
    np.testing.assert_almost_equal(r[2], 2*u.kpc)
    np.testing.assert_almost_equal(v[0], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[1], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[2], 20*u.km/u.s)

    # l = 90
    r,v = gal_xyz_to_hel_lbd([-8,2,0]*u.kpc,
                             [0,220,0]*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r[0], 90*u.deg)
    np.testing.assert_almost_equal(r[1], 0*u.deg)
    np.testing.assert_almost_equal(r[2], 2*u.kpc)
    np.testing.assert_almost_equal(v[0], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[1], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[2], 20*u.km/u.s)

    # l = 180
    r,v = gal_xyz_to_hel_lbd([-10,0,0]*u.kpc,
                             [-20,200,0]*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r[0], 180*u.deg)
    np.testing.assert_almost_equal(r[1], 0*u.deg)
    np.testing.assert_almost_equal(r[2], 2*u.kpc)
    np.testing.assert_almost_equal(v[0], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[1], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[2], 20*u.km/u.s)

    # l = 270
    r,v = gal_xyz_to_hel_lbd([-8,-2,0]*u.kpc,
                             [0,180,0]*u.km/u.s,
                             vlsr=[0.,0,0]*u.km/u.s,
                             vcirc=200*u.km/u.s)
    np.testing.assert_almost_equal(r[0], 270*u.deg)
    np.testing.assert_almost_equal(r[1], 0*u.deg)
    np.testing.assert_almost_equal(r[2], 2*u.kpc)
    np.testing.assert_almost_equal(v[0], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[1], 0*u.mas/u.yr)
    np.testing.assert_almost_equal(v[2], 20*u.km/u.s)

    print(r,v)

def test_sgr():

    d = np.genfromtxt(os.path.join(os.environ['STREAMSPATH'], 'data',
                                   'simulation', 'LM10', 'SgrTriax_DYN.dat'),
                      names=True)[:2]

    print(d.dtype.names)
    # l,b,D,mul,mub,vr = gc_to_hel(-d['xgc']*u.kpc, d['ygc']*u.kpc, d['zgc']*u.kpc,
    #                              -d['u']*u.km/u.s, d['v']*u.km/u.s, d['w']*u.km/u.s)
    (l,b,D),(mul,mub,vr) = gal_xyz_to_hel_lbd(r=np.vstack((-d['xgc'],d['ygc'],d['zgc']))*u.kpc,
                                              v=np.vstack((-d['u'],d['v'],d['w']))*u.km/u.s,
                                              vlsr=[0,0,0]*u.km/u.s,
                                              vcirc=220*u.km/u.s)

    print(l.degree, d['l'])
    print(b.degree, d['b'])
    print(D.value, d['dist'])

    print(mul.value*np.cos(b), d['mul'])
    print(mub.value, d['mub'])

def test_unitless():
    r = [15.6,11.23,-19.]*u.kpc
    v = [120.3,-64.51,15.]*u.km/u.s
    X = np.append(r.value, v.decompose(usys).value).T.copy()

    (l1,b1,d1),(mul1,mub1,vr1) = gal_xyz_to_hel_lbd(r, v=v,
                                    vlsr=[0.,0,0]*u.km/u.s,
                                    vcirc=220*u.km/u.s,
                                    xsun=-8.*u.kpc)

    l2,b2,d2,mul2,mub2,vr2 = _gc_to_hel(X).T

    np.testing.assert_almost_equal(l1.decompose(usys).value, l2)
    np.testing.assert_almost_equal(b1.decompose(usys).value, b2)
    np.testing.assert_almost_equal(d1.decompose(usys).value, d2)
    np.testing.assert_almost_equal(mul1.decompose(usys).value, mul2)
    np.testing.assert_almost_equal(mub1.decompose(usys).value, mub2)
    np.testing.assert_almost_equal(vr1.decompose(usys).value, vr2)

def test_roundtrip_unitless():
    np.random.seed(43)
    N = 100
    X = np.random.random((N,6))

    O = _gc_to_hel(X)
    X2 = _hel_to_gc(O)

    assert np.allclose(X, X2)