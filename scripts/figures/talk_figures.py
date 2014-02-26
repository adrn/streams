# coding: utf-8

""" A script for making figures for our Streams Paper 1 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from collections import defaultdict
import cPickle as pickle
import inspect

# Third-party
import astropy.units as u
from astropy.constants import G
from astropy.io.misc import fnpickle, fnunpickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc_context, rcParams, cm, animation
from matplotlib.patches import Rectangle, Ellipse, Circle
import scipy.optimize as so
from scipy.stats import chi2

from streams.integrate import LeapfrogIntegrator
import streams.io as io
from streams.io.sgr import SgrSimulation
from streams.observation import apparent_magnitude
from streams.observation.gaia import parallax_error, proper_motion_error
from streams.observation.rrlyrae import rrl_M_V, rrl_V_minus_I
from streams.plot import plot_point_cov
from streams.potential import LawMajewski2010
from streams.util import project_root
from streams import usys

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('axes', edgecolor='#333333', labelsize=24, labelweight=400, linewidth=2.0)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='Roboto', weight='light')

plot_path = "plots/talks/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

#sgr_color = '#0085c3'
sgr_color = '#666666'
#orp_color = '#867ac1'
def sgr():
    """ Visualize the Sgr stream model (from LM10) """
    #from mpl_toolkits.mplot3d import Axes3D

    nMilkyWay = 10000
    nSgr = None
    nOrp = None
    # nMilkyWay = 100
    # nSgr = 100
    # nOrp = 100

    lm10 = io.LM10Simulation()
    sgr = lm10.particles(N=nSgr, expr="(Pcol<8) & (abs(Lmflag)==1)")

    #kvj_orp = io.OrphanSimulation()
    #orp = kvj_orp.particles(N=nOrp)

    # radial scale length, vertical scale length, number
    # thin disk, thick disk
    mw_x = np.array([])
    mw_y = np.array([])
    mw_z = np.array([])
    for r0,h0,N in [(3.,0.3,int(nMilkyWay*49/50)), (3.,1.,int(nMilkyWay*1/50))]:
        r = np.random.exponential(scale=r0, size=N)
        phi = np.random.uniform(0., 2*np.pi, size=N)

        mw_x = np.append(mw_x, r * np.cos(phi))
        mw_y = np.append(mw_y, r * np.sin(phi))
        mw_z = np.append(mw_z, np.append(np.random.exponential(scale=h0, size=N//2), \
                                         -np.random.exponential(scale=h0, size=N//2)))

    rcparams = {}
    with rc_context(rc=rcparams):
        # 2d
        fig,axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.plot(mw_x[np.abs(mw_z) < 2], mw_z[np.abs(mw_z) < 2],
                  marker='.', alpha=0.1, color='k', linestyle='none', markersize=3.)
        # axes.plot(orp['x'].to(u.kpc).value, orp['z'].to(u.kpc).value,
        #           marker='.', alpha=0.2, color=orp_color, linestyle='none', markersize=4.)
        axes.plot(sgr['x'].to(u.kpc).value, sgr['z'].to(u.kpc).value,
                  marker='.', alpha=0.2, color=sgr_color, linestyle='none', markersize=4.)

        # scale markers
        axes.plot([35.,45],[15.,15.], linestyle='-', marker=None, color='k')
        #axes.plot([45,45],[15.,25.], linestyle='-', marker=None, color='k')

        axes.set_xlim(-90, 60)
        axes.set_ylim(-70, 80)
        axes.set_aspect('equal') # doesn't work in 3d

        # the sun
        axes.plot([-8.], [0.], linestyle='none', color='#e6550d', alpha=0.75,
                  marker='o', markersize=5, markeredgewidth=0.0)

        plt.axis('off')
        plt.tight_layout(pad=0.)
        fig.savefig(os.path.join(plot_path, "sgr_black.pdf"), bbox_inches='tight', transparent=True)

def nfw():
    def rho(r, Rs):
        rr = r/Rs
        return 1/(rr*(1+rr)**2)

    fig,ax = plt.subplots(1,1,figsize=(12,8))
    ax.plot(1,1)

    r = np.logspace(-2, 3, 50)
    ax.loglog(r, rho(r,1.), lw=7.)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.spines['bottom'].set_linewidth(5)
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_linewidth(5)
    ax.spines['left'].set_color('k')
    fig.savefig(os.path.join(plot_path, "nfw.pdf"), transparent=True)

def q_p(**kwargs):

    filename = os.path.join(plot_path, "q_p.pdf")
    fig,axes = plt.subplots(2,4,figsize=(14,7.5),
                            sharex=True, sharey=True)

    bins = np.linspace(0.,10,40)
    nparticles = 5000
    for kk,_m in enumerate(range(6,9+1)):
        mass = "2.5e{}".format(_m)
        m = float(mass)
        print(mass)

        sgr = SgrSimulation(mass)
        p = sgr.particles(N=nparticles, expr="(tub!=0)")#" & (tub<400)")
        tub = p.tub
        s = sgr.satellite()

        potential = LawMajewski2010()

        X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
        V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
        integrator = LeapfrogIntegrator(potential._acceleration_at,
                                        np.array(X), np.array(V),
                                        args=(X.shape[0], np.zeros_like(X)))
        ts, rs, vs = integrator.run(t1=sgr.t1, t2=sgr.t2, dt=-1.)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

        p_x = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
        s_x = np.array([s_orbit[jj,0] for jj in t_idx])

        #############################################
        # determine tail_bit
        diff = p_x-s_x
        norm_r = s_x[:,:3] / np.sqrt(np.sum(s_x[:,:3]**2, axis=-1))[:,np.newaxis]
        norm_diff_r = diff[:,:3] / np.sqrt(np.sum(diff[:,:3]**2, axis=-1))[:,np.newaxis]
        dot_prod_r = np.sum(norm_diff_r*norm_r, axis=-1)
        tail_bit = (dot_prod_r > 0.).astype(int)*2 - 1
        #############################################

        r_tide = potential._tidal_radius(m, s_orbit[...,:3])#*0.69336
        s_R_orbit = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
        a_pm = (s_R_orbit + r_tide*tail_bit) / s_R_orbit
        q = np.sqrt(np.sum((p_x[:,:3] - s_x[:,:3])**2,axis=-1))

        f = r_tide / s_R_orbit
        s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
        vdisp = s_V * f / 1.4
        p = np.sqrt(np.sum((p_x[:,3:] - s_x[...,3:])**2,axis=-1))

        fig,axes = plt.subplots(2,1,figsize=(10,6),sharex=True)

        axes[0].plot(tub, q, marker='.', alpha=0.5, color='#666666')
        axes[0].plot(ts, r_tide*1.4, linewidth=2., alpha=0.8, color='k',
                     linestyle='-', marker=None)
        axes[0].set_ylim(0., max(r_tide)*4)

        axes[1].plot(tub, (p*u.kpc/u.Myr).to(u.km/u.s).value,
                     marker='.', alpha=0.5, color='#666666')
        axes[1].plot(ts, (vdisp*u.kpc/u.Myr).to(u.km/u.s).value, color='k',
                     linewidth=2., alpha=0.75, linestyle='-', marker=None)

        M_enc = potential._enclosed_mass(s_R_orbit)
        #delta_E = 4/3.*G.decompose(usys).value**2*m*(M_enc / s_V)**2*r_tide**2/s_R_orbit**4
        delta_v2 = 4/3.*G.decompose(usys).value**2*(M_enc / s_V)**2*\
                        np.mean(r_tide**2)/s_R_orbit**4
        delta_v = (np.sqrt(2*delta_v2)*u.kpc/u.Myr).to(u.km/u.s).value

        axes[1].plot(ts, delta_v, linewidth=2., color='#2166AC',
                     alpha=0.75, linestyle='--', marker=None)

        axes[1].set_ylim(0., max((vdisp*u.kpc/u.Myr).to(u.km/u.s).value)*4)

        axes[0].set_xlim(min(ts), max(ts))

        fig.savefig(os.path.join(plot_path, "q_p_{}.png".format(mass)),
                    transparent=True)

    # fig.tight_layout()
    # fig.subplots_adjust(top=0.92, hspace=0.025, wspace=0.1)
    # fig.savefig(filename)

if __name__ == "__main__":
    #sgr()
    #nfw()
    q_p()