# coding: utf-8

""" A script for making figures for our streams paper 2 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle
import inspect

# Third-party
import astropy.units as u
from astropy.constants import G
from astropy.io.misc import fnpickle, fnunpickle
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc_context, rcParams, cm
from matplotlib.patches import Rectangle, Ellipse, Circle
import scipy.optimize as so
from scipy.stats import norm

from streams import usys
from streams.util import project_root
from streams.io.sgr import SgrSimulation, SgrSimulationDH
from streams.integrate import LeapfrogIntegrator
from streams.potential.lm10 import LawMajewski2010

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('axes', edgecolor='#444444', labelsize=24,
              labelweight=400, linewidth=2.0)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='Source Sans Pro')

SgrSimulation = SgrSimulationDH
# expr = "(tub!=0)"
# expr = "(tub!=0) & (tub<355) & (tub>94)" # for KVJ
expr = "(tub!=0) & (tub>675.)" # for DH

plot_path = "plots/paper2/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

ext = 'png'

def simulated_streams(**kwargs):

    filename = os.path.join(plot_path, "simulated_streams.{}".format(ext))
    fig,axes = plt.subplots(2,4,figsize=(14,7.5),
                            sharex=True, sharey=True)

    ticks = [-100,-50,0,50]
    alpha = 0.3
    rcparams = {'lines.linestyle' : 'none',
                'lines.marker' : '.',
                'lines.markersize' : 3}

    with rc_context(rc=rcparams):
        for ii,_m in enumerate(range(6,9+1)):
            mass = "2.5e{}".format(_m)
            print(mass)
            m = float(mass)

            sgr = SgrSimulation(mass)
            p = sgr.particles(N=10000)

            with rc_context(rc=rcparams):
                axes[0,ii].text(0.5, 1.04, r"$2.5\times10^{}M_\odot$".format(_m),
                       horizontalalignment='center',
                       fontsize=24,
                       transform=axes[0,ii].transAxes)

                axes[0,ii].plot(p["x"].value, p["y"].value,
                                alpha=alpha)
                axes[1,ii].plot(p["x"].value, p["z"].value,
                                alpha=alpha)
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
    fig.savefig(filename)

def potentials(**kwargs):

    filename = os.path.join(plot_path, "potentials.{}".format(ext))
    fig,axes = plt.subplots(2,4,figsize=(14,7))

    base_params = dict(q1=1., qz=1., q2=1., phi=0.)
    potentials = []
    potentials.append(LawMajewski2010(**base_params))

    pp = base_params.copy()
    pp['q1'] = 1.5
    potentials.append(LawMajewski2010(**pp))
    axes[0,1].text(0.5, 1.04, r"$q_1=1.5$",
                   horizontalalignment='center',
                   fontsize=20,
                   transform=axes[0,1].transAxes)

    pp = base_params.copy()
    pp['qz'] = 1.5
    potentials.append(LawMajewski2010(**pp))
    axes[0,2].text(0.5, 1.04, r"$q_z=1.5$",
                   horizontalalignment='center',
                   fontsize=20,
                   transform=axes[0,2].transAxes)

    pp = base_params.copy()
    pp['phi'] = 45*u.degree
    pp['q1'] = 1.5
    potentials.append(LawMajewski2010(**pp))
    axes[0,3].text(0.5, 1.04, r"$q_1=1.5$, $\phi=45^\circ$",
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

def Lpts(**kwargs):

    potential = LawMajewski2010()
    filename = os.path.join(plot_path, "Lpts_r.{}".format(ext))
    filename2 = os.path.join(plot_path, "Lpts_v.{}".format(ext))

    fig,axes = plt.subplots(2,4,figsize=(14,7.5),
                            sharex=True, sharey=True)
    fig2,axes2 = plt.subplots(2,4,figsize=(14,7.5),
                              sharex=True, sharey=True)

    bins = np.linspace(-3,3,50)
    nparticles = 2000
    for k,_m in enumerate(range(6,9+1)):
        mass = "2.5e{}".format(_m)
        m = float(mass)
        print(mass)

        sgr = SgrSimulation(mass)
        p = sgr.particles(N=nparticles, expr=expr)
        s = sgr.satellite()

        X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
        V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
        integrator = LeapfrogIntegrator(potential._acceleration_at,
                                        np.array(X), np.array(V),
                                        args=(X.shape[0], np.zeros_like(X)))
        ts, rs, vs = integrator.run(t1=sgr.t1, t2=sgr.t2, dt=-1.)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

        m_t = (-s.mdot*ts + s.m0)[:,np.newaxis]
        s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
        s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
        r_tide = sgr.true_potential._tidal_radius(m_t, s_orbit[...,:3])
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
        #circ = Circle((0,0), radius=1., fill=False, alpha=0.75,
        #              edgecolor='k', linestyle='solid')
        #axes[1,k].add_patch(circ)

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

        axes[0,k].text(0.5, 1.04, r"$2.5\times10^{}M_\odot$".format(_m),
                       horizontalalignment='center',
                       fontsize=24,
                       transform=axes[0,k].transAxes)

        axes2[0,k].text(0.5, 1.04, r"$2.5\times10^{}M_\odot$".format(_m),
                        horizontalalignment='center',
                        fontsize=24,
                        transform=axes2[0,k].transAxes)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    fig.savefig(filename)

    fig2.tight_layout()
    fig2.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    fig2.savefig(filename2)

def phasespace(**kwargs):

    potential = LawMajewski2010()
    filename = os.path.join(plot_path, "Lpts_rv.png")

    fig,axes = plt.subplots(3,4,figsize=(14,12),
                            sharex=True, sharey=True)

    bins = np.linspace(-3,3,50)
    nparticles = 2000
    for k,_m in enumerate(range(6,9+1)):
        mass = "2.5e{}".format(_m)
        m = float(mass)
        print(mass)

        sgr = SgrSimulation(mass)
        p = sgr.particles(N=nparticles, expr=expr)
        s = sgr.satellite()

        X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
        V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
        integrator = LeapfrogIntegrator(potential._acceleration_at,
                                        np.array(X), np.array(V),
                                        args=(X.shape[0], np.zeros_like(X)))
        ts, rs, vs = integrator.run(t1=sgr.t1, t2=sgr.t2, dt=-1.)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

        m_t = (-s.mdot*ts + s.m0)[:,np.newaxis]
        s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
        s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
        r_tide = sgr.true_potential._tidal_radius(m_t, s_orbit[...,:3])
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

def total_rv(**kwargs):

    potential = LawMajewski2010()
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

        sgr = SgrSimulation(mass)
        p = sgr.particles(N=nparticles, expr=expr)
        s = sgr.satellite()

        X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
        V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
        integrator = LeapfrogIntegrator(potential._acceleration_at,
                                        np.array(X), np.array(V),
                                        args=(X.shape[0], np.zeros_like(X)))
        ts, rs, vs = integrator.run(t1=sgr.t1, t2=sgr.t2, dt=-1.)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

        m_t = (-s.mdot*ts + s.m0)[:,np.newaxis]
        s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
        s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
        r_tide = sgr.true_potential._tidal_radius(m_t, s_orbit[...,:3])
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

def num_recombine(**kwargs):
    N_particles = kwargs.get('N', 1000)
    Dps_lim = 2.

    sgr_path = os.path.join(project_root, "data", "simulation", "Sgr")
    plot_path = os.path.join(project_root, "plots", "tests", "num_combine")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    data_file = os.path.join(plot_path, "frac_Dps{0}_N{1}.pickle".format(Dps_lim, N_particles))
    if args.overwrite and os.path.exists(data_file):
        os.remove(data_file)

    masses = ['2.5e{0}'.format(xx) for xx in range(6, 10)]
    canonical_errors = {"proper_motion_error_frac" : 1.,
                        "distance_error_percent" : 2.,
                        "radial_velocity_error" : 10.*u.km/u.s}
    error_fracs = [0.0001, 0.01, 0.1, 0.5, 1., 2.]

    if not os.path.exists(data_file):
        lm10 = LawMajewski2010()

        frac_recombined = np.zeros((len(masses), len(error_fracs), args.iter))
        for mm,mass in enumerate(masses):

            logger.info("Starting mass {0}...".format(mass))
            particles_today, satellite_today, time = mass_selector(mass)

            np.random.seed(args.seed)
            all_particles = particles_today(N=0)
            satellite = satellite_today()
            t1, t2 = time()
            for ii in range(args.iter):
                logger.debug("\t iteration {0}...".format(ii))
                true_particles = all_particles[np.random.randint(len(all_particles),
                                                                 size=N_particles)]

                for ee,error_frac in enumerate(error_fracs):
                    errors = canonical_errors.copy()
                    errors = dict([(k,v*error_frac) for k,v in errors.items()])
                    particles = add_uncertainties_to_particles(true_particles, **errors)

                    logger.debug("\t error frac.: {0}...".format(error_frac))

                    # integrate the orbits backwards, compute the minimum phase-space distance
                    integrator = SatelliteParticleIntegrator(lm10, satellite, particles)

                    import time
                    a = time.time()
                    s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
                    #s_orbit,p_orbits = integrator.run(timestep_func=timestep,
                    #                                  timestep_args=(lm10,),
                    #                                  resolution=5.,
                    #                                  t1=t1, t2=t2)
                    print(time.time()-a)
                    min_ps = minimum_distance_matrix(lm10, s_orbit, p_orbits)

                    D_ps = np.sqrt(np.sum(min_ps**2, axis=-1))
                    frac = np.sum(D_ps < Dps_lim) / N_particles
                    frac_recombined[mm,ee,ii] = frac

        fnpickle(frac_recombined, data_file)

    frac_recombined = fnunpickle(data_file)
    kwargs = dict(marker="o", linestyle="-", lw=1., alpha=0.5)
    colors = ["#D73027","#FC8D59","#E6F598","#99D594","#000000","#3288BD"][:frac_recombined.shape[1]]

    plt.figure(figsize=(12,8))
    for ee in range(frac_recombined.shape[1]):
        pp = kwargs.copy()
        pp['c'] = colors[ee]
        if error_fracs[ee] == 1.:
            pp['lw'] = 2.
            pp['c'] = 'k'

        for ii in range(frac_recombined.shape[2]):
            if ii == 0:
                plt.semilogx([float(m) for m in masses], frac_recombined[:,ee,ii],
                             label="{0:0.2f}".format(error_fracs[ee]), **pp)
            else:
                plt.semilogx([float(m) for m in masses], frac_recombined[:,ee,ii], **pp)

    plt.legend(title=r'$\times$ canonical', loc='lower right')
    plt.xticks([float(m) for m in masses], masses)
    plt.xlabel("Satellite mass [$M_\odot$]", fontsize=26)
    plt.ylabel("Fraction that recombine", fontsize=26)
    plt.tight_layout()
    plt.show()

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
    parser.add_argument("--kwargs", dest="kwargs", nargs="+", type=str,
                       help="kwargs passed in to whatever function you call.")

    args = parser.parse_args()
    try:
        kwargs = dict([tuple(k.split("=")) for k in args.kwargs])
    except TypeError:
        kwargs = dict()

    kwargs["overwrite"] = args.overwrite

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
    func(**kwargs)

