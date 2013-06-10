# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import cProfile
import pstats
import os
import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from streams.potential.lm10 import LawMajewski2010
from streams.data import lm10_particles, lm10_satellite, lm10_time, \
                         lm10_satellite_orbit, read_simulation
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.integrate.leapfrog import LeapfrogIntegrator

plot_path = "plots/tests/integrate/satellite_particles"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

potential = LawMajewski2010()
satellite = lm10_satellite()
particles = lm10_particles(N=100, expr="(Pcol>-1) & (abs(Lmflag)==1) & (dist < 80)")
t1,t2 = lm10_time()

class TestLM10(object):
    
    def test_against_orbit_file(self):
        integrator = SatelliteParticleIntegrator(potential, 
                                                 satellite, 
                                                 particles)
        
        s, p = integrator.run(t1=t1, t2=t2, dt=-1.)
        
        satellite_orbit = lm10_satellite_orbit()
        R_lm10 = np.sqrt(np.sum(satellite_orbit._r**2, axis=-1))
        V_lm10 = np.sqrt(np.sum(satellite_orbit._v**2, axis=-1))
        R_apw = np.sqrt(np.sum(s._r**2, axis=-1))
        V_apw = np.sqrt(np.sum(s._v**2, axis=-1))
        
        plt.plot(satellite_orbit.t, R_lm10, color='b', alpha=0.5, marker=None)
        plt.plot(s.t, R_apw, color='r', alpha=0.5, marker=None)
        plt.savefig(os.path.join(plot_path,"lm10_vs_apw_pos.png"))
        
        plt.clf()
        plt.plot(satellite_orbit.t, R_lm10, color='b', alpha=0.5, marker=None)
        plt.plot(s.t, R_apw, color='r', alpha=0.5, marker=None)
        plt.xlim(-2., 0.)
        plt.ylim(20., 20.5)
        plt.savefig(os.path.join(plot_path,"lm10_vs_apw_zoom_pos.png"))
        
        plt.clf()
        plt.plot(satellite_orbit.t, V_lm10, color='b', alpha=0.5, marker=None)
        plt.plot(s.t, V_apw, color='r', alpha=0.5, marker=None)
        plt.savefig(os.path.join(plot_path,"lm10_vs_apw_vel.png"))
        
        plt.clf()
        plt.plot(satellite_orbit.t, V_lm10, color='b', alpha=0.5, marker=None)
        plt.plot(s.t, V_apw, color='r', alpha=0.5, marker=None)
        plt.xlim(-2., 0.)
        plt.ylim(0.3, 0.35)
        plt.savefig(os.path.join(plot_path,"lm10_vs_apw_zoom_vel.png"))


"""

def timestep(r, v, potential, m_sat):
    R_tide = tidal_radius(potential, r[0], m_sat=m_sat)
    v_max = np.max(np.sqrt(np.sum(v[1:]**2,axis=-1)))
    return -(R_tide / v_max)

def test_sp():
    integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
    integrator.run(time_spec=dict(t1=t1, t2=t2, dt=-5.))
    
    integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
    integrator.run(time_spec=dict(t1=t1, t2=t2),
                   timestep_func=lambda r,v: -5.)
    
    import time
    a = time.time()
    for ii in range(10):
        s,p = integrator.run(time_spec=dict(t1=t1, t2=t2, dt=-5.))
    print( (time.time()-a) / 10.)

def profile_adaptive():
    res = 5.
    for ii in range(10):
        integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
        s_lores,p_lores = integrator.run(time_spec=dict(t1=t1, t2=t2),
                             timestep_func=timestep,
                             timestep_args=(lm10, satellite.m.value),
                             resolution=res)
    
def test_adaptive():
    
    res = 1.
    
    import time
    a = time.time()
    integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
    s_lores,p_lores = integrator.run(time_spec=dict(t1=t1, t2=t2),
                         timestep_func=timestep,
                         timestep_args=(lm10, satellite.m.value),
                         resolution=res)
    print("res: {0}, time: {1}, Nsteps: {2}".format(res, time.time()-a, len(s_lores.t)))
    
    a = time.time()
    sat_integrator = LeapfrogIntegrator(lm10._acceleration_at, satellite._r, satellite._v)
    par_integrator = LeapfrogIntegrator(lm10._acceleration_at, particles._r, particles._v)
    
    sat_trv = sat_integrator.run(time_spec=dict(t1=t1, t2=t2, dt=-1.))
    par_trv = par_integrator.run(time_spec=dict(t1=t1, t2=t2, dt=-1.))
    print("fixed timestep, time: {0}, Nsteps: {1}".format(time.time()-a, len(sat_trv[0])))
    
    fig,axes = plt.subplots(2,1,figsize=(10,10))
    axes[0].plot(sat_trv[0], sat_trv[1][:,0,2], marker=None, color='r', alpha=0.5)
    axes[0].plot(s_lores.t.value, s_lores._r[:,0,2], marker=None, color='b', alpha=0.5)
    
    axes[1].plot(sat_trv[0], sat_trv[1][:,0,2], marker='.', linestyle='none', color='r', alpha=0.5)
    axes[1].plot(s_lores.t.value, s_lores._r[:,0,2], marker='.', linestyle='none', color='b', alpha=0.5)
    
    fig.savefig(os.path.join(plot_path, "adaptive_sgr_orbit.png"))
    
    idx = np.random.randint(100, size=10)
    for jj in range(10):
        ii = idx[jj]
        fig,axes = plt.subplots(2,1,figsize=(10,10))
        axes[0].plot(par_trv[0], par_trv[1][:,ii,2], marker=None, color='r', alpha=0.5)
        axes[0].plot(p_lores.t.value, p_lores._r[:,ii,2], marker=None, color='b', alpha=0.5)
        
        axes[1].plot(par_trv[0], par_trv[1][:,ii,2], marker='.', linestyle='none', color='r', alpha=0.5)
        axes[1].plot(p_lores.t.value, p_lores._r[:,ii,2], marker='.', linestyle='none', color='b', alpha=0.5)
        
        fig.savefig(os.path.join(plot_path, "adaptive_particle_orbit_{0}.png".format(jj)))
    

if __name__ == "__main__":
    cProfile.run("profile_adaptive()", os.path.join(plot_path, "cprofiled"))
    p = pstats.Stats(os.path.join(plot_path, "cprofiled"))
    p.sort_stats('cumulative').print_stats(50)
"""