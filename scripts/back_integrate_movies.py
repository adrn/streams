# coding: utf-8

""" Create movies showing back-integration in correct and incorrect potentials """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import numpy as np

from streams.data import read_lm10
from streams.potential.lm10 import LawMajewski2010, true_params
from streams.integrate import leapfrog
from streams.dynamics import OrbitCollection
from streams.inference.backintegrate import relative_normalized_coordinates, \
                                            tidal_radius, \
                                            escape_velocity

ffmpeg_cmd = "ffmpeg -i {0} -r 12 -b 5000 -vcodec libx264 -vpre medium -b 3000k {1}"
plot_path = "plots/movies/back_integrate"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

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

def plot_3d_orbits(potential, s, p):
    grid = np.linspace(np.min(p._r)-2., np.max(p._r)+2., 200)*u.kpc
    fig,axes = potential.plot(ndim=3, grid=grid)
    axes[0,0].plot(s._r[:,0,0], s._r[:,0,1], color='w', alpha=0.25)
    axes[0,0].plot(p._r[:,:,0], p._r[:,:,1], color='w', alpha=0.01)
    
    axes[1,0].plot(s._r[:,0,0], s._r[:,0,2], color='w', alpha=0.25)
    axes[1,0].plot(p._r[:,:,0], p._r[:,:,2], color='w', alpha=0.01)
    
    axes[1,1].plot(s._r[:,0,1], s._r[:,0,2], color='w', alpha=0.25)
    axes[1,1].plot(p._r[:,:,1], p._r[:,:,2], color='w', alpha=0.01)
    
    return fig,axes

def plot_1d_orbits(potential, s, p):
    grid = np.linspace(np.min(p._r)-2., np.max(p._r)+2., 200)*u.kpc
    X1, X2 = np.meshgrid(grid.value,grid.value)
    
    r = np.array([np.zeros_like(X1.ravel()) for xx in range(3)])
    r[0] = X1.ravel()
    r[2] = X2.ravel()
    r = r.T*grid.unit
    
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    cs = ax.contourf(X1, X2, 
                     potential.value_at(r).value.reshape(X1.shape), 
                     cmap=cm.bone_r)
    
    ax.plot(s._r[:,0,0], s._r[:,0,2], color='w', alpha=0.25)
    ax.plot(p._r[:,:,0], p._r[:,:,2], color='w', alpha=0.05)
    
    return fig,ax

def plot_3d_animation(potential, s, p, filename=""):
    #grid = np.linspace(np.min(p._r)-2., np.max(p._r)+2., 200)*u.kpc
    grid = np.linspace(-81, 81, 200)*u.kpc
    fig = None
    
    idx = np.ones(p._r.shape[1]).astype(bool)
    R,V = relative_normalized_coordinates(potential, p, s)
    all_D_ps = np.sqrt(np.sum(R**2, axis=-1) + np.sum(V**2, axis=-1))
    r_tide = tidal_radius(potential, s)[:,:,np.newaxis]
    
    fig,axes = potential.plot(ndim=3, grid=grid)
    jj = 0
    for ii in range(0,len(t),10):
        D_ps = all_D_ps[ii]
        idx = idx & (D_ps > 2.8)
        
        offsets = p._r[ii]
        offsets[np.logical_not(idx)] = np.ones_like(offsets[np.logical_not(idx)])*10000.
        sat_r = s._r[ii,0]
        
        try:
            scatter_map[(0,0)].set_offsets(np.vstack((offsets[:,0],offsets[:,1])).T)
            scatter_map[(1,0)].set_offsets(np.vstack((offsets[:,0],offsets[:,2])).T)
            scatter_map[(1,1)].set_offsets(np.vstack((offsets[:,1],offsets[:,2])).T)
            
            scatter_map_sat[(0,0)].center = (sat_r[0], sat_r[1])
            scatter_map_sat[(0,0)].set_radius(r_tide[ii])
            scatter_map_sat[(1,0)].center = (sat_r[0], sat_r[2])
            scatter_map_sat[(1,0)].set_radius(r_tide[ii])
            scatter_map_sat[(1,1)].center = (sat_r[1], sat_r[2])
            scatter_map_sat[(1,1)].set_radius(r_tide[ii])
        except NameError:
            scatter_map = dict()
            scatter_map_sat = dict()
            
            c = Circle((sat_r[0], sat_r[1]), radius=r_tide[ii], color='r', alpha=0.5)
            axes[0,0].add_patch(c)
            scatter_map_sat[(0,0)] = c
            
            c = Circle((sat_r[0], sat_r[2]), radius=r_tide[ii], color='r', alpha=0.5)
            axes[1,0].add_patch(c)
            scatter_map_sat[(1,0)] = c
            
            c = Circle((sat_r[1], sat_r[2]), radius=r_tide[ii], color='r', alpha=0.5)
            axes[1,1].add_patch(c)
            scatter_map_sat[(1,1)] = c
            
            scatter_map[(0,0)] = axes[0,0].scatter(offsets[:,0], offsets[:,1], 
                                                   marker='.', color='w', 
                                                   alpha=0.2, s=5)
            scatter_map[(1,0)] = axes[1,0].scatter(offsets[:,0], offsets[:,2], 
                                                   marker='.', color='w', 
                                                   alpha=0.2, s=5)
            scatter_map[(1,1)] = axes[1,1].scatter(offsets[:,1], offsets[:,2], 
                                                   marker='.', color='w', 
                                                   alpha=0.2, s=5)
        
        axes[0,0].set_xlim(-81,81)
        axes[0,0].set_ylim(-81,81)
        fig.savefig(os.path.join(plot_path,"{0}{1:04d}.png".format(filename,jj)))
        jj += 1
    
    print("{0} unbound at end of run.".format(sum(idx)))

def xz_potential_contours(potential, grid):
    X1, X2 = np.meshgrid(grid.value,grid.value)
    
    r = np.array([np.zeros_like(X1.ravel()) for xx in range(3)])
    r[0] = X1.ravel()
    r[2] = X2.ravel()
    r = r.T*grid.unit
    
    fig,ax = plt.subplots(1,1,figsize=(12,12))
    cs = ax.contourf(X1, X2, 
                     potential.value_at(r).value.reshape(X1.shape), 
                     cmap=cm.Greys)
    
    return fig, ax

def plot_xz_animation(potential, s, p, filename=""):
    grid = np.linspace(-81, 81, 200)*u.kpc
    
    idx = np.ones(p._r.shape[1]).astype(bool)
    R,V = relative_normalized_coordinates(potential, p, s)
    all_D_ps = np.sqrt(np.sum(R**2, axis=-1) + np.sum(V**2, axis=-1))
    r_tide = tidal_radius(potential, s)[:,:,np.newaxis]
    
    fig,ax = xz_potential_contours(potential, grid)
    ax.set_frame_on(False)
    ax.set_xlabel("$X_{GC}$", color='w', fontsize=28)
    ax.set_ylabel("$Z_{GC}$", color='w', fontsize=28, rotation='horizontal')
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.06)
    
    jj = 0
    for ii in range(0,len(t),10):
        D_ps = all_D_ps[ii]
        idx = idx & (D_ps > 2.8)
        
        offsets = p._r[ii]
        offsets[np.logical_not(idx)] = np.ones_like(offsets[np.logical_not(idx)])*len(p._r)
        sat_r = s._r[ii,0]
        
        try:
            scatter_map[(1,0)].set_offsets(np.vstack((offsets[:,0],offsets[:,2])).T)
            scatter_map_sat[(1,0)].center = (sat_r[0], sat_r[2])
            scatter_map_sat[(1,0)].set_radius(r_tide[ii])
        except NameError:
            scatter_map = dict()
            scatter_map_sat = dict()
            
            c = Circle((sat_r[0], sat_r[2]), radius=r_tide[ii], 
                       facecolor='#CA0020', alpha=0.3, edgecolor='none')
            ax.add_patch(c)
            scatter_map_sat[(1,0)] = c
            
            scatter_map[(1,0)] = ax.scatter(offsets[:,0], offsets[:,2], 
                                            marker='.', color='#ABD9E9', 
                                            alpha=0.4, s=8)
            
        ax.set_xlim(-81,81)
        ax.set_ylim(-81,81)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.savefig(os.path.join(plot_path,"{0}{1:04d}.png".format(filename,jj)),
                    facecolor=(21/255.,21/255.,21/255.))
        jj += 1
    
    print("{0} unbound at end of run.".format(sum(idx)))

if __name__ == "__main__":
    
    N = 10000
    dt = 1.*u.Myr
    
    # Read in particles from Law & Majewski 2010 simulation of Sgr
    #expr="(Pcol>0) & (abs(Lmflag)==1)"
    t,satellite, particles = read_lm10(N=N, 
                                      expr="(Pcol > -1) & (Pcol < 7) & (abs(Lmflag) == 1)",
                                      dt=dt)
    
    # Define correct potential, and 10% wrong potential
    correct = LawMajewski2010()
    wrong = LawMajewski2010(q1=true_params["q1"]*1.05,
                            qz=true_params["qz"]*1.05,
                            phi=true_params["phi"]*1.05,
                            v_halo=true_params["v_halo"]*1.05)
    
    c_s_orbit, c_p_orbit = back_integrate(correct) # correct
    w_s_orbit, w_p_orbit = back_integrate(wrong) # wrong
        
    fig,axes = plot_3d_orbits(correct, c_s_orbit, c_p_orbit)
    fig.savefig(os.path.join(plot_path,"correct_orbits.png"))
    #plot_3d_animation(correct, c_s_orbit, c_p_orbit, filename="correct_")
    plot_xz_animation(correct, c_s_orbit, c_p_orbit, filename="correct_")
    
    fig,axes = plot_3d_orbits(wrong, w_s_orbit, w_p_orbit)
    fig.savefig(os.path.join(plot_path,"wrong_orbits.png"))
    #plot_3d_animation(wrong, w_s_orbit, w_p_orbit, filename="wrong_")
    plot_xz_animation(wrong, w_s_orbit, w_p_orbit, filename="wrong_")
    
    os.system(ffmpeg_cmd.format(os.path.join(plot_path, "correct_%4d.png"), 
                                os.path.join(plot_path, "correct.mp4")))
    
    os.system(ffmpeg_cmd.format(os.path.join(plot_path, "wrong_%4d.png"), 
                                os.path.join(plot_path, "wrong.mp4")))