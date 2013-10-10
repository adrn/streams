# coding: utf-8
"""
    Test energy conservation for LM10 potential -- see test_integrators.py 
    for others.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os,sys
import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from ...misc.units import UnitSystem
from ...potential.lm10 import LawMajewski2010
from ...io.lm10 import particles_today, satellite_today, time
from ..satellite_particles import satellite_particles_integrate

plot_path = "plots/tests/energy_conservation"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def energies(potential, orbit):
    E_kin = 0.5*np.sum(orbit._v**2, axis=-1)
    
    E_pot = np.zeros_like(orbit._r[:,:,0])
    for ii in range(orbit.nparticles):
        E_pot[:,ii] = potential._value_at(orbit._r[:,ii,:])    
    
    return E_kin, E_pot

def timestep(r, v, potential):
    """ From Dehnen & Read 2011 """
    r_i = np.sqrt(np.sum(r**2, axis=-1))
    m_encs = potential._enclosed_mass(r_i)
    dt = np.min(np.sqrt(r_i**3 / (potential._G * m_encs)))
    
    return -dt / 10.

def test_lm10():
    N = 100
    
    lm10 = LawMajewski2010()    
    particles = particles_today(N=N)
    satellite = satellite_today()
    t1, t2 = time()
    
    Nparticles = len(particles)
    acc = np.zeros((Nparticles+1,3)) # placeholder
    s_orbit,p_orbits = satellite_particles_integrate(satellite, particles,
                                                     potential,
                                                     potential_args=(Nparticles+1, acc), \
                                                     time_spec=dict(t1=t1, t2=t2, dt=-1.))
    
    print("{0} timesteps".format(len(s_orbit._t)))
    
    Ts,Vs = energies(lm10, s_orbit)    
    Tp,Vp = energies(lm10, p_orbits)
    
    fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,8))
    
    axes[0].plot(s_orbit._t, Ts[:,0], marker=None, alpha=0.5)
    for ii in range(N):
        axes[0].plot(s_orbit._t, Tp[:,ii], marker=None, alpha=0.1, color="#0571B0")
    axes[0].set_ylabel(r"$E_{kin}$")
    
    axes[1].plot(s_orbit._t, Vs, marker=None, alpha=0.5)
    for ii in range(N):
        axes[1].plot(s_orbit._t, Vp[:,ii], marker=None, alpha=0.1, color="#0571B0")
    axes[1].set_ylabel(r"$E_{pot}$")
    
    Es = (Ts + Vs)
    Ep = (Tp + Vp)
    axes[2].semilogy(s_orbit._t[1:], np.fabs((Es[1:]-Es[0])/Es[0]), marker=None, alpha=0.5)
    for ii in range(N):
        axes[2].semilogy(s_orbit._t[1:], np.fabs((Ep[1:,ii]-Ep[0,ii])/Ep[0,ii]), 
                         marker=None, alpha=0.1, color="#0571B0")
    axes[2].set_ylabel(r"$\Delta E/E$")
    axes[2].set_ylim(-1., 1.)
    
    fig.savefig(os.path.join(plot_path, "lm10.png"))