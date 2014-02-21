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
from matplotlib import rc_context, rcParams, cm
from matplotlib.patches import Rectangle, Ellipse
import scipy.optimize as so

from streams.integrate import LeapfrogIntegrator
import streams.io as io
from streams.observation import apparent_magnitude
from streams.observation.gaia import parallax_error, proper_motion_error
from streams.observation.rrlyrae import rrl_M_V, rrl_V_minus_I
from streams.plot import plot_point_cov
from streams.potential import LawMajewski2010
from streams.util import project_root

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

def rho(r, Rs):
    rr = r/Rs
    return 1/(rr*(1+rr)**2)

def nfw():

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

if __name__ == "__main__":
    #sgr()
    nfw()