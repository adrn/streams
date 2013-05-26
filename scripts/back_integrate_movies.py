# coding: utf-8

""" Create movies showing back-integration in correct and incorrect potentials """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from streams.data import read_lm10
from streams.potential.lm10 import LawMajewski2010, true_params
from streams.integrate import leapfrog
from streams.nbody import OrbitCollection
from streams.inference.backintegrate import relative_normalized_coordinates, \
                                            tidal_radius, \
                                            escape_velocity

ffmpeg_cmd = "ffmpeg -i {0} -r 12 -b 5000 -vcodec libx264 -vpre medium -b 3000k {1}"
plot_path = "plots/movies/back_integrate"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Read in particles from Law & Majewski 2010 simulation of Sgr
#expr="(Pcol>0) & (abs(Lmflag)==1)"
t,satellite, particles = read_lm10(N=5000, 
                                  expr="(abs(Lmflag)==1)",
                                  dt=1.*u.Myr)

def back_integrate(potential):
    xx,r,v = leapfrog(potential._acceleration_at, 
                      satellite._r, satellite._v,
                      t=t.value)
    satellite_orbit = OrbitCollection(t=t, 
                                      r=r*satellite.r.unit,
                                      v=v*satellite.v.unit,
                                      m=[2.5E8]*u.M_sun,
                                      units=[u.kpc, u.Myr, u.M_sun])
    
    xx,r,v = leapfrog(potential._acceleration_at, 
                      particles._r, particles._v,
                      t=t.value)
    particle_orbits = OrbitCollection(t=t, 
                                      r=r*particles.r.unit, 
                                      v=v*particles.v.unit, 
                                      m=np.ones(len(r))*u.M_sun,
                                      units=[u.kpc, u.Myr, u.M_sun])
    
    return satellite_orbit, particle_orbits

def plot_orbits(potential, s, p):
    grid = np.linspace(np.min(p._r)-2., np.max(p._r)+2., 200)*u.kpc
    fig,axes = potential.plot(ndim=3, grid=grid)
    axes[0,0].plot(s._r[:,0,0], s._r[:,0,1], color='w', alpha=0.25)
    axes[0,0].plot(p._r[:,:,0], p._r[:,:,1], color='w', alpha=0.05)
    
    axes[1,0].plot(s._r[:,0,0], s._r[:,0,2], color='w', alpha=0.25)
    axes[1,0].plot(p._r[:,:,0], p._r[:,:,2], color='w', alpha=0.05)
    
    axes[1,1].plot(s._r[:,0,1], s._r[:,0,2], color='w', alpha=0.25)
    axes[1,1].plot(p._r[:,:,1], p._r[:,:,2], color='w', alpha=0.05)
    
    return fig,axes

# Define correct potential, and 10% wrong potential
correct = LawMajewski2010()
wrong = LawMajewski2010(q1=true_params["q1"]*1.05,
                        qz=true_params["qz"]*1.05,
                        phi=true_params["phi"]*1.05,
                        v_halo=true_params["v_halo"]*1.05)

c_s_orbit, c_p_orbit = back_integrate(correct) # correct
w_s_orbit, w_p_orbit = back_integrate(wrong) # wrong

def plot_animation(potential, s, p, filename=""):
    grid = np.linspace(np.min(p._r)-2., np.max(p._r)+2., 200)*u.kpc
    fig = None
    
    idx = np.ones(p._r.shape[1]).astype(bool)
    R,V = relative_normalized_coordinates(potential, p, s)
    all_D_ps = np.sqrt(np.sum(R**2, axis=-1) + np.sum(V**2, axis=-1))
    r_tide = tidal_radius(potential, s)[:,:,np.newaxis]
    
    for ii in range(len(t)):
        if fig == None:
            fig,axes = potential.plot(ndim=3, grid=grid)
        else:
            [a.cla() for a in np.ravel(axes)]
            fig,axes = potential.plot(ndim=3, grid=grid, axes=axes)
        
        D_ps = all_D_ps[ii]
        idx = idx & (D_ps > 2.8)
        #eja = np.argmin(D_ps)
        #print(R[ii,eja], V[ii,eja], sum(idx))
        print(sum(idx))
        
        axes[0,0].set_xlim(-75,75)
        axes[0,0].set_ylim(-75,75)
        axes[0,0].scatter(s._r[ii,0,0], s._r[ii,0,1], marker='o', color='r', alpha=0.75)
        axes[0,0].scatter(p._r[ii,:,0][idx], p._r[ii,:,1][idx], marker='.', color='w', alpha=0.2, s=8)
        
        axes[1,0].scatter(s._r[ii,0,0], s._r[ii,0,2], marker='o', color='r', alpha=0.75)
        axes[1,0].scatter(p._r[ii,:,0][idx], p._r[ii,:,2][idx], marker='.', color='w', alpha=0.2, s=8)
        
        axes[1,1].scatter(s._r[ii,0,1], s._r[ii,0,2], marker='o', color='r', alpha=0.75)
        axes[1,1].scatter(p._r[ii,:,1][idx], p._r[ii,:,2][idx], marker='.', color='w', alpha=0.2, s=8)
        
        fig.savefig(os.path.join(plot_path,"{0}{1:04d}.png".format(filename,ii)))

fig,axes = plot_orbits(correct, c_s_orbit, c_p_orbit)
fig.savefig(os.path.join(plot_path,"correct_orbits.png"))
plot_animation(correct, c_s_orbit, c_p_orbit, filename="correct_")

fig,axes = plot_orbits(wrong, w_s_orbit, w_p_orbit)
fig.savefig(os.path.join(plot_path,"wrong_orbits.png"))
plot_animation(wrong, w_s_orbit, w_p_orbit, filename="wrong_")

os.system(ffmpeg_cmd.format(os.path.join(plot_path, "correct_%4d.png"), 
                            os.path.join(plot_path, "correct.mp4")))

os.system(ffmpeg_cmd.format(os.path.join(plot_path, "wrong_%4d.png"), 
                            os.path.join(plot_path, "wrong.mp4")))