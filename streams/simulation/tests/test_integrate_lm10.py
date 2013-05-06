# coding: utf-8
"""
    Test the Cython integrate code
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import time

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt

from .._integrate_lm10 import lm10_acceleration, leapfrog_lm10
from ...potential.lm10 import LawMajewski2010, halo_params
from ...integrate import leapfrog
from ...data import read_lm10
from ...simulation import TestParticleOrbit, generalized_variance

np.random.seed(5)
satellite_ic, particles = read_lm10(N=100, expr="(Pcol > 0) & (dist < 100)")
t = np.arange(satellite_ic.t1,
              satellite_ic.t2,
              -5.)*u.Myr

# LawMajewski2010 contains a disk, bulge, and logarithmic halo 
potential = LawMajewski2010(**halo_params)

def test_cython_vs_python1():
    r = np.random.random((100,3))
    
    Ntest = 10
    
    a = time.time()
    for ii in range(Ntest):
        acc1 = lm10_acceleration(r, len(r), 1.38, 1.36, 
                                1.692969, 0.124625659009).T
    cython = (time.time() - a) / float(Ntest)
    
    lm10 = LawMajewski2010()
    
    a = time.time()
    for ii in range(Ntest):
        acc2 = lm10.acceleration_at(r)
    pure_python = (time.time() - a) / float(Ntest)
    
    print("cython: {0}".format(cython))
    print("pure python: {0}".format(pure_python))
    
    assert cython < pure_python
    
def test_cython_vs_python2():
    r = np.random.random((100,3))
    v = np.random.random((100,3))
    t = np.arange(0, 7000, 10.)
    
    Ntest = 10
    
    a = time.time()
    for ii in range(Ntest):
        tt1,rr1,vv1 = leapfrog_lm10(r, v, t, 1.38, 1.36, 1.692969, 0.124625659009)
    cython = (time.time() - a) / Ntest
    
    lm10 = LawMajewski2010()
                              
    a = time.time()
    for ii in range(Ntest):
        tt2,rr2,vv2 = leapfrog(lm10.acceleration_at, r, v, t)
    pure_python = (time.time() - a) / Ntest
    
    plt.plot(tt1, rr1[:,0,0], color='r')
    plt.plot(tt1, rr2[:,0,0], color='b')
    plt.savefig("/var/www/scratch/cython_test.png")
    
    print("cython: {0}".format(cython))
    print("pure python: {0}".format(pure_python))
    assert cython < pure_python

def test_compare_satellite_orbit():
        
    py_satellite_orbit = satellite_ic.integrate(potential, t)
    py_particle_orbits = particles.integrate(potential, t)
       
    sat_and_particles = particles.add_particle(satellite_ic)
    orbits = sat_and_particles._lm10_integrate(t, halo_params["q1"],
                                               halo_params["qz"],
                                               halo_params["phi"].to(u.radian).value,
                                               halo_params["v_halo"].to(u.kpc/u.Myr).value)
    
    p_t = orbits.t
    p_r = orbits.r[:,:-1,:]
    p_v = orbits.v[:,:-1,:]
    cy_particle_orbits = TestParticleOrbit(p_t, p_r, p_v)
    
    s_t = orbits.t
    s_r = orbits.r[:,-1,:][:,np.newaxis,:]
    s_v = orbits.v[:,-1,:][:,np.newaxis,:]
    cy_satellite_orbit = TestParticleOrbit(s_t, s_r, s_v) 
    
    fig,axes = plt.subplots(3,1)
    for ii in range(3):
        axes[ii].plot(t.value, py_satellite_orbit.r[:,0,ii].value, 'r-')
        axes[ii].plot(t.value, cy_satellite_orbit.r[:,0,ii].value, 'b-')
    
    
    py_var = generalized_variance(potential, py_particle_orbits, py_satellite_orbit)
    cy_var = generalized_variance(potential, cy_particle_orbits, cy_satellite_orbit)
    
    plt.savefig("/var/www/scratch/cython_test.png")

def test_likelihood():
    from streams.inference import ln_likelihood, ln_likelihood_lm10, \
                                  ln_posterior, ln_posterior_lm10
    
    p = halo_params["q1"], halo_params["qz"], \
        halo_params["phi"].value, halo_params["v_halo"].to(u.km/u.s).value
    p_names = ["q1", "qz", "phi", "v_halo"]
    
    py_like = ln_likelihood(p, p_names, particles, satellite_ic, t)
    cy_like = ln_likelihood_lm10(p, particles, satellite_ic, t)
    
    print("python likelihood: ", py_like)
    print("cython likelihood: ", cy_like)
    