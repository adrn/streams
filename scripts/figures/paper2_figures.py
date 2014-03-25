# coding: utf-8

""" A script for making figures for our streams paper 2 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle
import inspect
from collections import OrderedDict

# Third-party
import astropy.units as u
from astropy.constants import G
from astropy.io.misc import fnpickle, fnunpickle
import h5py
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc_context, rcParams, cm
from matplotlib.patches import Rectangle, Ellipse, Circle
import scipy.optimize as so
from scipy.stats import norm
import triangle

from streams import usys
from streams.util import streamspath, _unit_transform, _label_map
from streams.coordinates.frame import galactocentric
from streams.io.sgr import SgrSimulation
from streams.io import read_hdf5, read_config
from streams.inference import StreamModel
from streams.integrate import LeapfrogIntegrator
from streams.potential.lm10 import LawMajewski2010

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('axes', edgecolor='#444444', labelsize=24,
              labelweight=400, linewidth=1.5)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='Source Sans Pro')

# expr = "(tub!=0)"
expr = "(tub!=0) & (tub>1800) & (tub<5500)"
sgr_path = 'sgr_nfw/M2.5e+0{}'
snapfile = 'SNAP113'
# sgr_path = 'sgr_plummer/2.5e{}'
# snapfile = 'SNAP'

plot_path = os.path.join(streamspath, "plots/paper2/")
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

ext = 'png'
grid_figsize = (14,7.5)

def simulated_streams():

    filename = os.path.join(plot_path, "simulated_streams.{}".format(ext))
    fig,axes = plt.subplots(2,4,figsize=grid_figsize,
                            sharex=True, sharey=True)

    ticks = [-100,-50,0,50]
    alphas = [0.2, 0.27, 0.34, 0.4]
    rcparams = {'lines.linestyle' : 'none',
                'lines.marker' : ','}

    with rc_context(rc=rcparams):
        for ii,_m in enumerate(range(6,9+1)):
            alpha = alphas[ii]
            mass = "2.5e{}".format(_m)
            print(mass)
            m = float(mass)

            data_filename = os.path.join(streamspath, "data", "observed_particles",
                                         "2.5e{}.hdf5".format(_m))
            cfg_filename = os.path.join(streamspath, "config", "exp1_{}.yml".format(_m))
            data = read_hdf5(data_filename)
            true_particles = data["true_particles"].to_frame(galactocentric)
            config = read_config(cfg_filename)
            idx = config['particle_idx']

            sgr = SgrSimulation(sgr_path.format(_m),snapfile)
            p = sgr.particles()
            p_bound = sgr.particles(expr="tub==0")

            axes[0,ii].text(0.5, 1.05, r"$2.5\times10^{}M_\odot$".format(_m),
                   horizontalalignment='center',
                   fontsize=24,
                   transform=axes[0,ii].transAxes)

            axes[0,ii].plot(p["x"].value, p["y"].value,
                            alpha=alpha, rasterized=True, color='#555555')
            axes[1,ii].plot(p["x"].value, p["z"].value,
                            alpha=alpha, rasterized=True, color='#555555')

            if _m == 8:
                axes[0,ii].plot(true_particles["x"].value[idx],
                             true_particles["y"].value[idx],
                             marker='+', markeredgewidth=1.5,
                             markersize=8, alpha=0.9, color='k')
                axes[1,ii].plot(true_particles["x"].value[idx],
                             true_particles["z"].value[idx],
                             marker='+', markeredgewidth=1.5,
                             markersize=8, alpha=0.9, color='k')
            axes[1,ii].set_xticks(ticks)
            axes[1,ii].set_xlabel("$X$ [kpc]")

    axes[0,0].set_ylabel("$Y$ [kpc]")
    axes[1,0].set_ylabel("$Z$ [kpc]")

    axes[0,0].set_yticks(ticks)
    axes[1,0].set_yticks(ticks)
    axes[-1,-1].set_xlim(-110,75)
    axes[-1,-1].set_ylim(-110,75)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    fig.savefig(filename, dpi=200)

def potentials():

    filename = os.path.join(plot_path, "potentials.{}".format(ext))
    fig,axes = plt.subplots(2,4,figsize=grid_figsize)

    base_params = dict(q1=1., qz=1., q2=1., phi=0.)
    potentials = []
    potentials.append(LawMajewski2010(**base_params))

    pp = base_params.copy()
    pp['q1'] = 1.5
    potentials.append(LawMajewski2010(**pp))
    axes[0,1].text(0.5, 1.05, r"$q_1=1.5$",
                   horizontalalignment='center',
                   fontsize=20,
                   transform=axes[0,1].transAxes)

    pp = base_params.copy()
    pp['qz'] = 1.5
    potentials.append(LawMajewski2010(**pp))
    axes[0,2].text(0.5, 1.05, r"$q_z=1.5$",
                   horizontalalignment='center',
                   fontsize=20,
                   transform=axes[0,2].transAxes)

    pp = base_params.copy()
    pp['phi'] = 45*u.degree
    pp['q1'] = 1.5
    potentials.append(LawMajewski2010(**pp))
    axes[0,3].text(0.5, 1.05, r"$q_1=1.5$, $\phi=45^\circ$",
                   horizontalalignment='center',
                   fontsize=20,
                   transform=axes[0,3].transAxes)

    grid = np.linspace(-75, 75, 250)
    X1, X2 = np.meshgrid(grid,grid)

    # top row:
    r = np.array([np.zeros_like(X1.ravel()).tolist() \
                   for xx in range(3)])
    r[0] = X1.ravel()
    r[1] = X2.ravel()

    levels = None
    for ii,potential in enumerate(potentials):
        axes[0,ii].set_xticks([-50,0,50])
        axes[0,ii].set_yticks([-50,0,50])

        Z = potential._value_at(r.T).reshape(X1.shape)
        if levels is None:
            cs = axes[0,ii].contourf(X1, X2, Z, cmap=cm.Blues_r)
            levels = cs.levels
        else:
            cs = axes[0,ii].contourf(X1, X2, Z, cmap=cm.Blues_r, levels=levels)

        if ii > 0:
            axes[0,ii].set_yticklabels([])

        axes[0,ii].set_xticklabels([])
        axes[0,ii].set_aspect('equal', 'box')

    # bottom row:
    r = np.array([np.zeros_like(X1.ravel()).tolist() \
                   for xx in range(3)])
    r[0] = X1.ravel()
    r[2] = X2.ravel()
    for ii,potential in enumerate(potentials):
        axes[1,ii].set_xticks([-50,0,50])
        axes[1,ii].set_yticks([-50,0,50])

        Z = potential._value_at(r.T).reshape(X1.shape)
        if levels is None:
            cs = axes[1,ii].contourf(X1, X2, Z, cmap=cm.Blues_r)
            levels = cs.levels
        else:
            cs = axes[1,ii].contourf(X1, X2, Z, cmap=cm.Blues_r, levels=levels)

        if ii > 0:
            axes[1,ii].set_yticklabels([])
        axes[1,ii].set_aspect('equal', 'box')
        axes[1,ii].set_xlabel("$X$ [kpc]")

    axes[0,0].set_ylabel("$Y$ [kpc]")
    axes[1,0].set_ylabel("$Z$ [kpc]")

    fig.tight_layout(pad=1.5, h_pad=0.)
    fig.savefig(filename)

def Lpts():

    potential = LawMajewski2010()
    filename = os.path.join(plot_path, "Lpts_r.{}".format(ext))
    filename2 = os.path.join(plot_path, "Lpts_v.{}".format(ext))

    fig,axes = plt.subplots(2,4,figsize=grid_figsize,
                            sharex=True, sharey=True)
    fig2,axes2 = plt.subplots(2,4,figsize=grid_figsize,
                              sharex=True, sharey=True)

    bins = np.linspace(-3,3,50)
    nparticles = 2000
    for k,_m in enumerate(range(6,9+1)):
        mass = "2.5e{}".format(_m)
        m = float(mass)
        print(mass)

        sgr = SgrSimulation(sgr_path.format(_m),snapfile)
        p = sgr.particles(n=nparticles, expr=expr)
        s = sgr.satellite()

        X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
        V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
        integrator = LeapfrogIntegrator(sgr.potential._acceleration_at,
                                        np.array(X), np.array(V),
                                        args=(X.shape[0], np.zeros_like(X)))
        ts, rs, vs = integrator.run(t1=sgr.t1, t2=sgr.t2, dt=-1.)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

        m_t = (-s.mdot*ts + s.m0)[:,np.newaxis]
        s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
        s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
        r_tide = sgr.potential._tidal_radius(m_t, s_orbit[...,:3])
        v_disp = s_V * r_tide / s_R

        # cartesian basis to project into
        x_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
        _y_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
        z_hat = np.cross(x_hat, _y_hat)
        y_hat = -np.cross(x_hat, z_hat)

        # translate to satellite position
        rel_orbits = p_orbits - s_orbit
        rel_pos = rel_orbits[...,:3]
        rel_vel = rel_orbits[...,3:]

        # project onto each
        X = np.sum(rel_pos * x_hat, axis=-1)
        Y = np.sum(rel_pos * y_hat, axis=-1)
        Z = np.sum(rel_pos * z_hat, axis=-1)

        VX = np.sum(rel_vel * x_hat, axis=-1)
        VY = np.sum(rel_vel * y_hat, axis=-1)
        VZ = np.sum(rel_vel * z_hat, axis=-1)
        VV = np.sqrt(VX**2+VY**2+VZ**2)

        _tcross = r_tide / np.sqrt(G.decompose(usys).value*m/r_tide)
        for ii,jj in enumerate(t_idx):
            #tcross = r_tide[jj,0] / _v[jj,ii]
            tcross = _tcross[jj]
            bnd = int(tcross / 2)

            ix1,ix2 = jj-bnd, jj+bnd
            if ix1 < 0: ix1 = 0
            if ix2 > max(sgr.t1,sgr.t2): ix2 = -1
            axes[0,k].plot(X[jj-bnd:jj+bnd,ii]/r_tide[jj-bnd:jj+bnd,0],
                           Y[jj-bnd:jj+bnd,ii]/r_tide[jj-bnd:jj+bnd,0],
                           linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)
            axes[1,k].plot(X[jj-bnd:jj+bnd,ii]/r_tide[jj-bnd:jj+bnd,0],
                           Z[jj-bnd:jj+bnd,ii]/r_tide[jj-bnd:jj+bnd,0],
                           linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)

        circ = Circle((0,0), radius=1., fill=False, alpha=0.75,
                      edgecolor='k', linestyle='solid')
        axes[0,k].add_patch(circ)
        circ = Circle((0,0), radius=1., fill=False, alpha=0.75,
                     edgecolor='k', linestyle='solid')
        axes[1,k].add_patch(circ)

        axes[0,k].axhline(0., color='k', alpha=0.75)
        axes[1,k].axhline(0., color='k', alpha=0.75)

        axes[0,k].set_xlim(-5,5)
        axes[0,k].set_ylim(axes[0,k].get_xlim())

        axes[1,k].set_xlabel(r"$x_1$")

        if k == 0:
            axes[0,k].set_ylabel(r"$x_2$")
            axes[1,k].set_ylabel(r"$x_3$")

        _tcross = r_tide / np.sqrt(G.decompose(usys).value*m/r_tide)
        for ii,jj in enumerate(t_idx):
            #tcross = r_tide[jj,0] / _v[jj,ii]
            tcross = _tcross[jj]
            bnd = int(tcross / 2)

            ix1,ix2 = jj-bnd, jj+bnd
            if ix1 < 0: ix1 = 0
            if ix2 > max(sgr.t1,sgr.t2): ix2 = -1
            axes2[0,k].plot(VX[jj-bnd:jj+bnd,ii]/v_disp[jj-bnd:jj+bnd,0],
                            VY[jj-bnd:jj+bnd,ii]/v_disp[jj-bnd:jj+bnd,0],
                            linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)
            axes2[1,k].plot(VX[jj-bnd:jj+bnd,ii]/v_disp[jj-bnd:jj+bnd,0],
                            VZ[jj-bnd:jj+bnd,ii]/v_disp[jj-bnd:jj+bnd,0],
                            linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)

        circ = Circle((0,0), radius=1., fill=False, alpha=0.75,
                      edgecolor='k', linestyle='solid')
        axes2[0,k].add_patch(circ)
        circ = Circle((0,0), radius=1., fill=False, alpha=0.75,
                      edgecolor='k', linestyle='solid')
        axes2[1,k].add_patch(circ)

        axes2[0,k].axhline(0., color='k', alpha=0.75)
        axes2[1,k].axhline(0., color='k', alpha=0.75)

        axes2[1,k].set_xlim(-5,5)
        axes2[1,k].set_ylim(axes2[1,k].get_xlim())

        axes2[1,k].set_xlabel(r"$v_{x_1}$")

        if k == 0:
            axes2[0,k].set_ylabel(r"$v_{x_2}$")
            axes2[1,k].set_ylabel(r"$v_{x_3}$")

        axes[0,k].text(0.5, 1.05, r"$2.5\times10^{}M_\odot$".format(_m),
                       horizontalalignment='center',
                       fontsize=24,
                       transform=axes[0,k].transAxes)

        axes2[0,k].text(0.5, 1.05, r"$2.5\times10^{}M_\odot$".format(_m),
                        horizontalalignment='center',
                        fontsize=24,
                        transform=axes2[0,k].transAxes)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    fig.savefig(filename)

    fig2.tight_layout()
    fig2.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    fig2.savefig(filename2)

def phasespace():

    filename = os.path.join(plot_path, "Lpts_rv.png")
    fig,axes = plt.subplots(3,4,figsize=(14,12),
                            sharex=True, sharey=True)

    bins = np.linspace(-3,3,50)
    nparticles = 2000
    for k,_m in enumerate(range(6,9+1)):
        mass = "2.5e{}".format(_m)
        m = float(mass)
        print(mass)

        sgr = SgrSimulation(sgr_path.format(_m),snapfile)
        p = sgr.particles(n=nparticles, expr=expr)
        s = sgr.satellite()

        X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
        V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
        integrator = LeapfrogIntegrator(sgr.potential._acceleration_at,
                                        np.array(X), np.array(V),
                                        args=(X.shape[0], np.zeros_like(X)))
        ts, rs, vs = integrator.run(t1=sgr.t1, t2=sgr.t2, dt=-1.)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

        m_t = (-s.mdot*ts + s.m0)[:,np.newaxis]
        s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
        s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
        r_tide = sgr.potential._tidal_radius(m_t, s_orbit[...,:3])
        v_disp = s_V * r_tide / s_R

        # cartesian basis to project into
        x_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
        _y_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
        z_hat = np.cross(x_hat, _y_hat)
        y_hat = -np.cross(x_hat, z_hat)

        # translate to satellite position
        rel_orbits = p_orbits - s_orbit
        rel_pos = rel_orbits[...,:3]
        rel_vel = rel_orbits[...,3:]

        # project onto each
        X = np.sum(rel_pos * x_hat, axis=-1)
        Y = np.sum(rel_pos * y_hat, axis=-1)
        Z = np.sum(rel_pos * z_hat, axis=-1)

        VX = np.sum(rel_vel * x_hat, axis=-1)
        VY = np.sum(rel_vel * y_hat, axis=-1)
        VZ = np.sum(rel_vel * z_hat, axis=-1)
        VV = np.sqrt(VX**2+VY**2+VZ**2)

        _tcross = r_tide / np.sqrt(G.decompose(usys).value*m/r_tide)
        for ii,jj in enumerate(t_idx):
            #tcross = r_tide[jj,0] / _v[jj,ii]
            tcross = _tcross[jj]
            bnd = int(tcross / 2)

            ix1,ix2 = jj-bnd, jj+bnd
            if ix1 < 0: ix1 = 0
            if ix2 > max(sgr.t1,sgr.t2): ix2 = -1
            axes[0,k].plot(X[jj-bnd:jj+bnd,ii]/r_tide[jj-bnd:jj+bnd,0],
                           VX[jj-bnd:jj+bnd,ii]/v_disp[jj-bnd:jj+bnd,0],
                           linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)

            axes[1,k].plot(Y[jj-bnd:jj+bnd,ii]/r_tide[jj-bnd:jj+bnd,0],
                           VY[jj-bnd:jj+bnd,ii]/v_disp[jj-bnd:jj+bnd,0],
                           linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)

            axes[2,k].plot(Z[jj-bnd:jj+bnd,ii]/r_tide[jj-bnd:jj+bnd,0],
                           VZ[jj-bnd:jj+bnd,ii]/v_disp[jj-bnd:jj+bnd,0],
                           linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)

        axes[0,k].set_xlim(-5,5)
        axes[0,k].set_ylim(axes[0,k].get_xlim())

        # axes[1,k].set_xlabel(r"$x_1$")

        # if k == 0:
        #     axes[0,k].set_ylabel(r"$x_2$")
        #     axes[1,k].set_ylabel(r"$x_3$")

        axes[0,k].set_title(r"$2.5\times10^{}M_\odot$".format(_m))

    fig.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    fig.savefig(filename)

def total_rv():

    filenamer = os.path.join(plot_path, "rel_r.png")
    filenamev = os.path.join(plot_path, "rel_v.png")

    figr,axesr = plt.subplots(4,1,figsize=(10,14),
                              sharex=True)
    figv,axesv = plt.subplots(4,1,figsize=(10,14),
                              sharex=True)

    nparticles = 2000
    for k,_m in enumerate(range(6,9+1)):
        mass = "2.5e{}".format(_m)
        m = float(mass)
        print(mass)

        sgr = SgrSimulation(sgr_path.format(_m),snapfile)
        p = sgr.particles(n=nparticles, expr=expr)
        s = sgr.satellite()

        X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
        V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
        integrator = LeapfrogIntegrator(sgr.potential._acceleration_at,
                                        np.array(X), np.array(V),
                                        args=(X.shape[0], np.zeros_like(X)))
        ts, rs, vs = integrator.run(t1=sgr.t1, t2=sgr.t2, dt=-1.)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

        m_t = (-s.mdot*ts + s.m0)[:,np.newaxis]
        s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
        s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
        r_tide = sgr.potential._tidal_radius(m_t, s_orbit[...,:3])
        v_disp = s_V * r_tide / s_R

        # cartesian basis to project into
        x_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
        _y_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
        z_hat = np.cross(x_hat, _y_hat)
        y_hat = -np.cross(x_hat, z_hat)

        # translate to satellite position
        rel_orbits = p_orbits - s_orbit
        rel_pos = rel_orbits[...,:3]
        rel_vel = rel_orbits[...,3:]

        # project onto each
        X = np.sum(rel_pos * x_hat, axis=-1)
        Y = np.sum(rel_pos * y_hat, axis=-1)
        Z = np.sum(rel_pos * z_hat, axis=-1)
        RR = np.sqrt(X**2 + Y**2 + Z**2)

        VX = np.sum(rel_vel * x_hat, axis=-1)
        VY = np.sum(rel_vel * y_hat, axis=-1)
        VZ = np.sum(rel_vel * z_hat, axis=-1)
        VV = (np.sqrt(VX**2 + VY**2 + VZ**2)*u.kpc/u.Myr).to(u.km/u.s).value
        v_disp = (v_disp*u.kpc/u.Myr).to(u.km/u.s).value

        _tcross = r_tide / np.sqrt(G.decompose(usys).value*m/r_tide)
        for ii,jj in enumerate(t_idx):
            #tcross = r_tide[jj,0] / _v[jj,ii]
            tcross = _tcross[jj]
            bnd = int(tcross / 2)

            ix1,ix2 = jj-bnd, jj+bnd
            if ix1 < 0: ix1 = 0
            if ix2 > max(sgr.t1,sgr.t2): ix2 = -1
            axesr[k].plot(ts[ix1:ix2],
                          RR[ix1:ix2,ii],
                          linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)

            axesv[k].plot(ts[ix1:ix2],
                          VV[ix1:ix2,ii],
                          linestyle='-', alpha=0.1, marker=None, color='#555555', zorder=-1)

        axesr[k].plot(ts, r_tide*2., marker=None)

        axesr[k].set_xlim(ts.min(), ts.max())
        axesv[k].set_xlim(ts.min(), ts.max())

        axesr[k].set_ylim(0,max(r_tide)*7)
        axesv[k].set_ylim(0,max(v_disp)*7)

        # axes[1,k].set_xlabel(r"$x_1$")

        # if k == 0:
        #     axes[0,k].set_ylabel(r"$x_2$")
        #     axes[1,k].set_ylabel(r"$x_3$")

        axesr[k].text(3000, max(r_tide)*5, r"$2.5\times10^{}M_\odot$".format(_m))
        axesv[k].text(3000, max(v_disp)*5, r"$2.5\times10^{}M_\odot$".format(_m))

    axesr[-1].set_xlabel("time [Myr]")
    axesv[-1].set_xlabel("time [Myr]")

    figr.suptitle("Relative distance", fontsize=26)
    figr.tight_layout()
    figr.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    figr.savefig(filenamer)

    figv.suptitle("Relative velocity", fontsize=26)
    figv.tight_layout()
    figv.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    figv.savefig(filenamev)

def trace_plots():
    cfg_filename = os.path.join(streamspath, "config", "exp1_8.yml")
    config = read_config(cfg_filename)
    model = StreamModel.from_config(config)

    hdf5_filename = os.path.join(streamspath, "plots", "hotfoot", "exper1_8", "inference.hdf5")
    with h5py.File(hdf5_filename, "r") as f:
        chain = f["chain"].value
        flatchain = f["flatchain"].value
        acceptance_fraction = f["acceptance_fraction"].value
        p0 = f["p0"].value
        acor = f["acor"].value

    labels = ["$q_1$", "$q_z$", r"$\phi$", "$v_h$", r"$\alpha$"]
    bounds = [(1.3,1.5),(1.25,1.45),(85,115),(118,132),(0.5,2.5)]
    ticks = [(1.35,1.4,1.45),(1.3,1.35,1.4),(90,100,110),(120,125,130),(1.,1.5,2.)]

    # plot individual walkers
    fig,axes = plt.subplots(5,1,figsize=(8.5,11),sharex=True)

    k = 0
    for gname,group in model.parameters.items():
        for pname,p in group.items():
            thischain = _unit_transform[pname](chain[...,k])

            for ii in range(config['walkers']):
                axes.flat[k].plot(thischain[ii,5000:],
                                  alpha=0.1, marker=None,
                                  drawstyle='steps', color='k')
            #axes.flat[k].set_ylabel(labels[k], rotation='horizontal')
            axes[k].text(-0.02, 0.5, labels[k],
                         horizontalalignment='right',
                         fontsize=22,
                         transform=axes[k].transAxes)

            if pname == "phi":
                axes[k].text(1.07, 0.475, "deg",
                         horizontalalignment='left',
                         fontsize=18,
                         transform=axes[k].transAxes)

            elif pname == "v_halo":
                axes[k].text(1.07, 0.475, "km/s",
                         horizontalalignment='left',
                         fontsize=18,
                         transform=axes[k].transAxes)

            axes[k].text(0.25, 0.1, r"$t_{\rm acor}$=" + "{}".format(int(acor[k])),
                         horizontalalignment='right',
                         fontsize=18,
                         transform=axes[k].transAxes)

            axes.flat[k].set_yticks(ticks[k])
            axes.flat[k].set_ylim(bounds[k])
            axes.flat[k].yaxis.tick_right()
            #axes.flat[k].yaxis.set_label_position("right")

            k += 1

    axes.flat[-1].set_xlabel("Step number")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.04, left=0.14, right=0.86)
    fig.savefig(os.path.join(plot_path, "mcmc_trace.{}".format(ext)))

def exp1_posterior():
    cfg_filename = os.path.join(streamspath, "config", "exp1_8.yml")
    config = read_config(cfg_filename)
    model = StreamModel.from_config(config)

    hdf5_filename = os.path.join(streamspath, "plots", "hotfoot", "exper1_8", "inference.hdf5")
    with h5py.File(hdf5_filename, "r") as f:
        chain = f["chain"].value
        #flatchain = f["flatchain"].value
        acceptance_fraction = f["acceptance_fraction"].value
        p0 = f["p0"].value
        acor = f["acor"].value

    _flatchain = np.vstack(chain[:,2500::int(np.median(acor))])
    flatchain = np.zeros_like(_flatchain)

    params = OrderedDict(model.parameters['potential'].items() + \
                         model.parameters['satellite'].items())
    labels = ["$q_1$", "$q_z$", r"$\phi$ [deg]", "$v_h$ [km/s]", r"$\alpha$"]

    truths = []
    bounds = []
    for ii,p in enumerate(params.values()):
        if p.name == 'alpha':
            truths.append(np.nan)
            bounds.append((1., 2.0))
            flatchain[:,ii] = _unit_transform[p.name](_flatchain[:,ii])
            continue

        truth = _unit_transform[p.name](p.truth)
        truths.append(truth)
        bounds.append((0.95*truth, 1.05*truth))
        flatchain[:,ii] = _unit_transform[p.name](_flatchain[:,ii])

    fig = triangle.corner(flatchain, plot_datapoints=False,
                          truths=truths, extents=bounds, labels=labels)
    fig.savefig(os.path.join(plot_path, "exp1_posterior.png"))

def exp_posteriors(exp_num, slicey=-5000):
    cfg_filename = os.path.join(streamspath, "config", "exp{}.yml".format(exp_num))
    config = read_config(cfg_filename)
    model = StreamModel.from_config(config)

    hdf5_filename = os.path.join(streamspath, "plots", "hotfoot",
                                "exper{}".format(exp_num), "inference.hdf5")
    # hdf5_filename = os.path.join(streamspath, "plots", "infer_potential",
    #                              "exper{}".format(exp_num), "inference.hdf5")
    with h5py.File(hdf5_filename, "r") as f:
        chain = f["chain"].value
        acceptance_fraction = f["acceptance_fraction"].value
        _p0 = f["p0"].value
        acor = f["acor"].value

    print("median acor: {}".format(int(np.median(acor))))

    if slicey is None:
        slicey = slice(-5000,None,int(np.median(acor)))
    else:
        slicey = slice(slicey,None,int(np.median(acor)))

    _flatchain = np.vstack(chain[acceptance_fraction>0.03,slicey])
    d = model.label_flatchain(_flatchain)
    p0 = model.label_flatchain(_p0)

    # Potential
    this_flatchain = np.zeros((_flatchain.shape[0], len(d["potential"])))
    this_p0 = np.zeros((_p0.shape[0], this_flatchain.shape[1]))
    truths = []
    bounds = []
    labels = []
    for ii,pname in enumerate(d["potential"].keys()):
        this_flatchain[:,ii] = _unit_transform[pname](np.squeeze(d["potential"][pname]))
        this_p0[:,ii] = _unit_transform[pname](np.squeeze(p0["potential"][pname]))

        p = model.parameters["potential"][pname]
        truth = _unit_transform[pname](p.truth)
        truths.append(truth)
        bounds.append((0.75*truth, 1.25*truth))
        labels.append(_label_map[pname])

    fig = triangle.corner(this_flatchain, plot_datapoints=False,
                          truths=truths, extents=bounds, labels=labels)
    fig.savefig(os.path.join(plot_path, "exp{}_potential.{}".format(exp_num, ext)))

    # Particle
    p_idx = 5
    this_flatchain = np.zeros((_flatchain.shape[0], len(d["particles"])))
    this_p0 = np.zeros((_p0.shape[0], this_flatchain.shape[1]))
    truths = []
    bounds = []
    labels = []
    for ii,pname in enumerate(d["particles"].keys()):
        this_flatchain[:,ii] = _unit_transform[pname](d["particles"][pname][:,p_idx])
        this_p0[:,ii] = _unit_transform[pname](p0["particles"][pname][:,p_idx])

        p = model.parameters["particles"][pname]
        truth = _unit_transform[pname](p.truth[p_idx])
        truths.append(truth)

        if pname == "tub":
            bounds.append((model.lnpargs[1], model.lnpargs[0]))
        else:
            sig = model.particles.errors[pname].value[p_idx]
            mu = model.particles[pname].value[p_idx]
            bounds.append((mu-3*sig, mu+3*sig))

        labels.append(_label_map[pname])

    # HACK
    bounds = [(32.5,41), (1.75,2.4), (-1.45,-1.0), (-85,-40), bounds[-1]]
    fig = triangle.corner(this_flatchain, plot_datapoints=False,
                          truths=truths, labels=labels, extents=bounds)
    fig.savefig(os.path.join(plot_path, "exp{}_particle.{}".format(exp_num, ext)))

    # Satellite
    this_flatchain = np.zeros((_flatchain.shape[0], len(d["satellite"])))
    this_p0 = np.zeros((_p0.shape[0], this_flatchain.shape[1]))
    truths = []
    bounds = []
    labels = []

    #for ii,pname in enumerate(keys):
    for ii,pname in enumerate(d["satellite"].keys()):
        this_flatchain[:,ii] = _unit_transform[pname](d["satellite"][pname][:,0])
        this_p0[:,ii] = _unit_transform[pname](p0["satellite"][pname][:,0])

        p = model.parameters["satellite"][pname]
        truth = _unit_transform[pname](p.truth)

        if pname == "alpha":
            bounds.append((1., 2.5))
            truths.append(np.nan)
        else:
            truths.append(truth)
            sig = model.satellite.errors[pname].value[0]
            mu = model.satellite[pname].value[0]
            bounds.append((mu-3*sig, mu+3*sig))

        labels.append(_label_map[pname])

    # HACK
    bounds = [(28,33), (-2.1,-1.6), (1.3,1.8), (125,185), bounds[-1]]
    if len(d["satellite"]) > len(bounds):
        bounds = [(0,10), (-20,5)] + bounds

    #bounds = None
    fig = triangle.corner(this_flatchain, plot_datapoints=False,
                          truths=truths, labels=labels, extents=bounds)
    fig.savefig(os.path.join(plot_path, "exp{}_satellite.{}".format(exp_num, ext)))

def exp2_posteriors(slicey=-10000):
    exp_posteriors(2)

def exp3_posteriors():
    exp_posteriors(3, slicey=-10000)

def exp4_posteriors():
    exp_posteriors(4, slicey=-7500)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Create logger
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-l", "--list", action="store_true", dest="list",
                        default=False, help="List all functions")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        dest="overwrite",  default=False,
                        help="Overwrite existing files.")
    parser.add_argument("-f", "--function", dest="function", type=str,
                        help="The name of the function to execute.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    def _print_funcs():
        fs = inspect.getmembers(sys.modules[__name__],
                                lambda member: inspect.isfunction(member) and member.__module__ == __name__ and not member.__name__.startswith("_"))
        print("\n".join([f[0] for f in fs]))

    if args.list:
        print("="*79)
        _print_funcs()
        print("="*79)
        sys.exit(0)

    if args.function is None:
        print ("You must specify a function name! Use -l to get the list "
               "of functions.")
        sys.exit(1)

    func = getattr(sys.modules[__name__], args.__dict__.get("function"))
    func()

