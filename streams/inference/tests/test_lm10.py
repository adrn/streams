# coding: utf-8
""" """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time
import cProfile
import pstats

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm

from streams.inference.lm10 import ln_likelihood, old_ln_likelihood
from streams.nbody import ParticleCollection
from streams.potential.lm10 import LawMajewski2010, true_params
from streams.data import lm10_particles, lm10_satellite, lm10_time

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

potential = LawMajewski2010()

def test_cprofile_time():
    for ii in range(10):
        test_time_likelihood()

def time_likelihood_func():
    a = time.time()
    for ii in range(10):
        test_time_likelihood()
    print((time.time()-a) / 10., "seconds per call")

np.random.seed(42)
t1,t2 = lm10_time()
satellite = lm10_satellite()
particles = lm10_particles(N=100, expr="(Pcol > -1) & (abs(Lmflag)==1) & (dist<75)")

def test_time_likelihood():
    
    param_names = ["q1", "qz", "v_halo", "phi"]  
    
    resolution = 2.
    
    p = [1.2, 1.2, 0.121, 1.6912]
    ln_likelihood(p, param_names, particles, satellite, 2.5E8, 
                  t1, t2, resolution)
    
    #p = [1.2, 1.2, 0.125, 1.6912]
    #print(ln_likelihood(p, param_names, particles, satellite, 2.5E8, 
    #                    t1, t2, resolution))

def test_likelihood_max():
    
    v_halos = np.linspace(0.124, 0.126, 10)
    
    old_ln_likelihood
    old_likelihoods = []
    for v_halo in v_halos:
        L = old_ln_likelihood([v_halo], ['v_halo'], particles, satellite, 2.5E8, 
                              t1, t2)
        old_likelihoods.append(L)
    
    for res in np.linspace(2., 4., 10):
        likelihoods = []
        for v_halo in v_halos:
            L = ln_likelihood([v_halo], ['v_halo'], particles, satellite, 2.5E8, 
                              t1, t2, res)
            print(res, v_halo, L)
            likelihoods.append(L)
        
        plt.clf()
        plt.plot((v_halos*u.kpc/u.Myr).to(u.km/u.s).value, likelihoods)
        plt.plot((v_halos*u.kpc/u.Myr).to(u.km/u.s).value, old_likelihoods, color='b')
        plt.axvline(true_params['v_halo'].to(u.km/u.s).value, color='r')
        plt.savefig(os.path.join(plot_path, "res{0}.png".format(res)))

def test_optimize():
    from scipy.optimize import fmin_bfgs
    
    res = 3.
    
    def apw_ln_likelihood(*args, **kwargs): 
        if args[0] < 0.1 or args[0] > 0.2:
            return 1E6
        print(args[0])
        return -ln_likelihood(*args, **kwargs)
    
    output = fmin_bfgs(apw_ln_likelihood, x0=[0.12],
                       args=(['v_halo'], particles, satellite, 
                             2.5E8, t1, t2, res))
    print(output)

if __name__ == "__main__":
    cProfile.run("test_cprofile_time()", os.path.join(plot_path, "cprofiled"))
    
    p = pstats.Stats(os.path.join(plot_path, "cprofiled"))
    p.sort_stats('cumulative').print_stats(50)
    
    time_likelihood_func()

"""

def test_energy_conserve():
    N = 100
    r = np.tile([5.,0.,0.], N).reshape(N,3) \
            * np.random.uniform(0.5, 1.5, size=(N,1)) \
            * u.kpc
    
    v = np.tile([0.,200.,0.], N).reshape(N,3) \
            * np.random.uniform(0.5, 1.5, size=(N,1)) \
            * u.km/u.s
    
    particles = ParticleCollection(r=r,
                                   v=v,
                                   m=np.zeros(N)*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    t = np.arange(0., 6000., 1.)*u.Myr
    
    tt,rr,vv = leapfrog(potential._acceleration_at, 
                        particles._r, particles._v, t=t.value)
    
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    ax.axis('off')
    #ax.scatter(r[:,0], r[:,1], marker='.', color='k', edgecolor='none')
    
    R = np.sum(rr[0]**2, axis=-1)
    scale = lambda R2: (R2 - R.min()) / (R.max()-R.min())
    
    cmap = cm.get_cmap('RdYlBu')
    for ii in range(N):
        this_R = rr[0,ii,0]**2 + rr[0,ii,2]**2
        ax.plot(rr[:,ii,0], rr[:,ii,1], color=cmap(scale(this_R)), alpha=0.07, lw=2.)
        
    ax.set_xlim(-15,15)
    ax.set_ylim(-15,15)
    fig.savefig(os.path.join(plot_path, "particles_one_orbit.png"),
                facecolor="#444444")

def test_compare_likelihood():
    satellite = lm10_satellite()
    particles = lm10_particles(N=100)
    t1,t2 = lm10_time()
    
    p = [1.2, 1.2, 0.121, 1.6912]
    param_names = ["q1", "qz", "v_halo", "phi"]
    a = time.time()
    l1 = ln_likelihood(p, param_names, particles, satellite, 2.5E8*u.M_sun, t1=t1, t2=t2, resolution=3.)
    print("l1: {0} ({1} seconds)".format(l1, time.time()-a))
    
    p = [1.38, 1.36, (121.858*u.km/u.s), 1.692969*u.radian]
    param_names = ["q1", "qz", "v_halo", "phi"]
    a = time.time()
    l2 = ln_likelihood(p, param_names, particles, satellite, 2.5E8*u.M_sun, t1=t1, t2=t2, resolution=3.)
    print("l2: {0} ({1} seconds)".format(l2, time.time()-a))
    
    assert l2 > l1

def test_timestep_energy():
    satellite = lm10_satellite()
    particles = lm10_particles(N=1000)
    t1,t2 = lm10_time()
    
    plt.figure(figsize=(12,12))
    c = 'krgb'
    for ii,dt in enumerate([1., 5., 10., 20.]):
        t = np.arange(t1, t2, -dt)
        tt,rr,vv = leapfrog(potential._acceleration_at, 
                            satellite._r, satellite._v, t=t)
        
        pot_energy = []
        for r in rr:
            pot_energy.append(potential._value_at(r))
        pot_energy = np.array(pot_energy)
        kin_energy = 0.5*np.sum(vv**2, axis=-1)
        
        tot_E = pot_energy+kin_energy
        delta_E = (tot_E[1:] - tot_E[0]) / tot_E[0]
        
        plt.subplot(211)
        plt.plot(tt[1:], delta_E, c=c[ii], label="dt={0}".format(dt))
        
        R = np.sqrt(np.sum(rr**2, axis=-1))
        plt.subplot(212)
        plt.plot(tt, R, c=c[ii])
    
    plt.subplot(211)
    plt.legend()
    plt.ylabel("$\Delta E/E$")
    
    plt.subplot(212)
    plt.xlabel("Time [Myr]")
    plt.ylabel("Sgr Orbital Radius $R$")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path,"timestep_energy_conserv.png"))

if __name__ == "__main__":
    cProfile.run("test_cprofile_time()", os.path.join(plot_path, "cprofiled"))
    
    p = pstats.Stats(os.path.join(plot_path, "cprofiled"))
    p.sort_stats('cumulative').print_stats(50)
"""