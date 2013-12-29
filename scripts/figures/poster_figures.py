# coding: utf-8

""" Figures for AAS223 poster """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle
import inspect

# Third-party
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle
import numpy as np
import matplotlib
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
from streams.potential import LawMajewski2010
from streams.util import project_root

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('axes', edgecolor='#333333', labelsize=24, labelweight=400, linewidth=2.0)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='Roboto', weight='light')
matplotlib.rc('savefig', transparent=True)

plot_path = "plots/poster/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

sgr_color = '#0085c3'
orp_color = '#867ac1'
def fig1(**kwargs):
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

    kvj_orp = io.OrphanSimulation()
    orp = kvj_orp.particles(N=nOrp)

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
        # 3d
        # fig,axes = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='3d'))
        # axes.plot(mw_x[np.abs(mw_z) < 2], mw_y[np.abs(mw_z) < 2], mw_z[np.abs(mw_z) < 2],
        #           marker='.', alpha=0.1, color='k', linestyle='none', markersize=3.)
        # axes.plot(orp['x'].to(u.kpc).value, orp['y'].to(u.kpc).value, orp['z'].to(u.kpc).value,
        #           marker='.', alpha=0.2, color=orp_color, linestyle='none', markersize=4.)
        # axes.plot(sgr['x'].to(u.kpc).value, sgr['y'].to(u.kpc).value, sgr['z'].to(u.kpc).value,
        #           marker='.', alpha=0.2, color=sgr_color, linestyle='none', markersize=4.)
        # axes.plot([-75.,-65.],[0.,0.],[40.,40.], linestyle='-', marker=None, color='k')
        # axes.set_xlim(-90, 60)
        # axes.set_zlim(-70, 80)
        # axes.view_init(elev=0., azim=270.)

        # 2d
        fig,axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.plot(mw_x[np.abs(mw_z) < 2], mw_z[np.abs(mw_z) < 2],
                  marker='.', alpha=0.1, color='k', linestyle='none', markersize=3.)
        axes.plot(orp['x'].to(u.kpc).value, orp['z'].to(u.kpc).value,
                  marker='.', alpha=0.2, color=orp_color, linestyle='none', markersize=4.)
        axes.plot(sgr['x'].to(u.kpc).value, sgr['z'].to(u.kpc).value,
                  marker='.', alpha=0.2, color=sgr_color, linestyle='none', markersize=4.)
        axes.plot([40.,50],[30.,30.], linestyle='-', marker=None, color='k')
        axes.plot([45,45],[25.,35.], linestyle='-', marker=None, color='k')
        axes.set_xlim(-90, 60)
        axes.set_ylim(-70, 80)
        axes.set_aspect('equal') # doesn't work in 3d

        # the sun
        axes.plot([-8.], [0.], linestyle='none', color='#ffff55', alpha=1.,
                  marker='o', markersize=6, markeredgewidth=0.0)

        plt.axis('off')
        plt.tight_layout(pad=0.)
        fig.savefig(os.path.join(plot_path, "fig1.pdf"), bbox_inches='tight')

def fig2(**kwargs):
    """ Visualize the observational errors from Gaia and Spitzer, along with
        dispersion and distance scale of Sgr and Orphan.
    """

    rcparams = {'lines.linestyle' : '-',
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'axes.facecolor' : '#ffffff',
                'xtick.major.size' : 16,
                'xtick.major.width' : 1.5,
                'xtick.minor.size' : 0,
                'ytick.major.size' : 16,
                'ytick.major.width' : 1.5,
                'ytick.minor.size' : 0}

    with rc_context(rc=rcparams):
        fig,axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Distance from 1kpc to ~100kpc
        D = np.logspace(0., 2., 50)*u.kpc

        # Sample metallicities from: http://arxiv.org/pdf/1211.7073v1.pdf
        fe_h = -1.67

        # Johnson/Cousins (V - I_C) color for RR Lyrae at *minimum*
        # Guldenschuh et al. (2005 PASP 117, 721), pg. 725
        rrl_V_minus_I = np.random.normal(0.579, 0.006)

        # Compute the apparent magnitude as a function of distance
        M_V = rrl_M_V(fe_h=fe_h)[0]
        m_V = apparent_magnitude(M_V, D)

        # Distance error
        dp = parallax_error(m_V, rrl_V_minus_I).to(u.arcsecond).value
        dD = D.to(u.pc).value**2 * dp * u.pc

        # Velocity error
        dpm = proper_motion_error(m_V, rrl_V_minus_I)
        dVtan = (dpm*D).to(u.km*u.radian/u.s).value
        #dVtan = (D*dpm + dD*1.*u.mas/u.yr).to(u.km*u.radian/u.s).value

        # Plot Gaia distance errors
        fracDist = (dD/D).decompose().value
        fracDist[fracDist > 0.1] = 0.1
        axes[0].loglog(D.to(u.kpc).value, fracDist, color='#000000', alpha=1., linewidth=4.)

        # Plot tangential velocity errors
        axes[1].loglog(D.to(u.kpc).value, dVtan, color='#000000', alpha=1., linewidth=4.)

        # Add spitzer 2% line to distance plot
        axes[0].axhline(0.02, linestyle='--', linewidth=3, color='k', alpha=0.75)

    # Now add rectangles for Sgr, Orphan
    sgr_d = Rectangle((10., 0.15), 60., 0.15,
                      color=sgr_color, alpha=0.75, label='Sgr stream')
    axes[0].add_patch(sgr_d)

    # From fig. 3 in http://mnras.oxfordjournals.org/content/389/3/1391.full.pdf+html
    orp_d = Rectangle((10., 0.03), 35., 0.03,
                      color=orp_color, alpha=0.75, label='Orp stream')
    axes[0].add_patch(orp_d)

    # Dispersion from Majewski 2004: 10 km/s
    sgr_v = Rectangle((10., 10), 60., 1., alpha=0.75, color=sgr_color)
    axes[1].add_patch(sgr_v)

    orp_v = Rectangle((10., 8.), 35., 1., alpha=0.75, color=orp_color)
    axes[1].add_patch(orp_v)

    axes[0].set_ylim(0.003, 1.3)
    axes[0].set_xlim(1, 100)
    axes[1].set_ylim(0.06, 130)

    axes[0].set_yticklabels([])
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels([])

    axes[0].yaxis.tick_right()
    axes[0].xaxis.tick_bottom()
    axes[1].yaxis.tick_right()
    axes[1].xaxis.tick_bottom()

    plt.tight_layout()
    #fig.subplots_adjust(hspace=0.05)
    fig.savefig(os.path.join(plot_path, "fig2.pdf"))

def fig3(**kwargs):
    """ Plot the PSD for 10 stars vs. back-integration time. """

    seed = int(kwargs.get("seed", 999))
    nPlot= 10
    nIntegrate = 1000
    dt = -1.

    # Read in the LM10 data
    np.random.seed(seed)
    lm10 = io.LM10Simulation()
    particles = lm10.particles(N=nIntegrate,
                               expr="(Pcol>-1) & (abs(Lmflag)==1) & (Pcol < 8)")
    satellite = lm10.satellite()

    # array of starting 6d positions
    gc = np.vstack((satellite._X,particles._X)).copy()
    acc = np.zeros_like(gc[:,:3])

    true_potential = LawMajewski2010()
    true_params = dict([(k,v.truth) for k,v in true_potential.parameters.items()])

    wrong_params = true_params.copy()
    wrong_params['v_halo'] = 1.25*wrong_params['v_halo']
    wrong_potential = LawMajewski2010(**wrong_params)

    sat_R = list()
    D_pses = list()
    ts = list()
    for potential in [true_potential, wrong_potential]:
        integrator = LeapfrogIntegrator(potential._acceleration_at,
                                        np.array(gc[:,:3]), np.array(gc[:,3:]),
                                        args=(gc.shape[0], acc))
        times, rs, vs = integrator.run(t1=lm10.t1, t2=lm10.t2, dt=dt)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T

        sat_var = np.zeros((len(times),6))
        sat_var[:,:3] = potential._tidal_radius(2.5e8, s_orbit[...,:3])*1.26
        sat_var[:,3:] += 0.02179966
        cov = (sat_var**2)[:,np.newaxis]

        D_ps = np.sqrt(np.sum((p_orbits - s_orbit)**2 / cov, axis=-1))
        D_pses.append(D_ps)

        sat_R.append(np.sqrt(np.sum(s_orbit[:,0,:3]**2, axis=-1)))
        ts.append(times)

    rcparams = {'xtick.major.size' : 16,
                'xtick.major.width' : 1.5}
    with rc_context(rc=rcparams):
        fig = plt.figure(figsize=(12.5,7))
        gs = GridSpec(2,4)

        axes = [plt.subplot(gs[0,:3]), plt.subplot(gs[1,:3]),
                plt.subplot(gs[0,3]), plt.subplot(gs[1,3])]
        axes[0].axhline(1.4, linestyle='--', color='#444444')
        axes[1].axhline(1.4, linestyle='--', color='#444444')

        for ii in range(nPlot):
            for jj in range(2):
                d = D_pses[jj][:,ii]
                sR = sat_R[jj]
                axes[jj].semilogy(ts[jj]/1000, d, alpha=0.4, color=sgr_color, linewidth=2.)
                axes[jj].semilogy(ts[jj][np.argmin(d)]/1000, np.min(d), marker='|',
                                  markeredgewidth=4, markeredgecolor='k', color='k',
                                  alpha=0.9, markersize=25)

        axes[0].set_ylim(0.6,20)
        axes[0].set_xlim(-6.1, 0.)
        axes[1].set_ylim(axes[0].get_ylim())
        axes[1].set_xlim(axes[0].get_xlim())
        axes[0].tick_params(axis='y', which='both', length=0., labelleft='off')
        axes[1].tick_params(axis='y', which='both', length=0., labelleft='off')

        # vertical histograms of D_ps values
        ylim = axes[0].get_ylim()
        bins = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), 50)
        n,xx,patches = axes[2].hist(np.min(D_pses[0], axis=0), bins=bins,
                                    orientation='horizontal', histtype='step',
                                    linewidth=2.)
        n,xx,patches = axes[3].hist(np.min(D_pses[1], axis=0), bins=bins,
                                    orientation='horizontal', histtype='step',
                                    linewidth=2.)
        axes[2].set_yscale('log')
        axes[3].set_yscale('log')
        axes[2].axis('off')
        axes[3].axis('off')

        axes[2].set_ylim(axes[0].get_ylim())
        axes[3].set_ylim(axes[1].get_ylim())
        axes[2].set_xlim(right=1.05*axes[2].get_xlim()[1])
        axes[3].set_xlim(right=1.05*axes[3].get_xlim()[1])

        axes[1].xaxis.tick_bottom()
        axes[1].set_xticklabels([])
        axes[0].xaxis.set_visible(False)

    fig.subplots_adjust(hspace=0.02, wspace=0., top=0.98, bottom=0.02, left=0.02, right=0.98)
    fig.savefig(os.path.join(plot_path, "fig3.pdf"))

def fig4(**kwargs):
    data_file = os.path.join(project_root, "plots", "hotfoot",
                             "SMASH_aspen", "all_best_parameters.pickle")

    with open(data_file) as f:
        data = pickle.load(f)

    rcparams = {'axes.linewidth' : 3.,
                'xtick.major.size' : 8.}
    with rc_context(rc=rcparams):
        fig,axes = plt.subplots(2, 2, figsize=(12,5), sharex='col', sharey='row')

    y_param = 'v_halo'
    x_params = ['q1', 'qz', 'phi']

    data['phi'] = (data['phi']*u.radian).to(u.degree).value
    _true_params['phi'] = (_true_params['phi']*u.radian).to(u.degree).value
    data['v_halo'] = (data['v_halo']*u.kpc/u.Myr).to(u.km/u.s).value
    _true_params['v_halo'] = (_true_params['v_halo']*u.kpc/u.Myr).to(u.km/u.s).value

    style = dict()
    style['q1'] = dict(ticks=[1.3, 1.38, 1.46],
                       lims=(1.27, 1.49))
    style['qz'] = dict(ticks=[1.28, 1.36, 1.44],
                       lims=(1.25, 1.47))
    style['phi'] = dict(ticks=[92, 97, 102],
                        lims=(90, 104))
    style['v_halo'] = dict(ticks=[115, 122, 129],
                           lims=(110, 134))

    for ii,x_param in enumerate(x_params):

        true_x = _true_params[x_param]
        true_y = _true_params[y_param]
        xdata = np.array(data[x_param])
        ydata = np.array(data[y_param])

        axes[ii].axhline(true_y, linewidth=2, color='#ABDDA4', alpha=0.75, zorder=-1)
        axes[ii].axvline(true_x, linewidth=2, color='#ABDDA4', alpha=0.75, zorder=-1)

        points = np.vstack([xdata, ydata]).T
        plot_point_cov(points, nstd=2, ax=axes[ii], alpha=0.25, color='#777777')
        plot_point_cov(points, nstd=1, ax=axes[ii], alpha=0.5, color='#777777')
        plot_point_cov(points, nstd=2, ax=axes[ii], color='#000000', fill=False)
        plot_point_cov(points, nstd=1, ax=axes[ii], color='#000000', fill=False)

        axes[ii].plot(xdata, ydata, marker='.', markersize=7, alpha=0.75,
                      color='#2B83BA', linestyle='none')
        axes[ii].set_xlim(style[x_param]['lims'])
        axes[ii].set_ylim(style[y_param]['lims'])

        axes[ii].yaxis.tick_left()
        axes[ii].set_xticks(style[x_param]['ticks'])
        axes[ii].xaxis.tick_bottom()

    axes[0].set_ylabel(r"$v_{\rm halo}$",
                       fontsize=26, rotation='horizontal')

    axes[0].set_yticks(style[y_param]['ticks'])
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    axes[0].set_xlabel(r"$q_1$", fontsize=26, rotation='horizontal')
    axes[1].set_xlabel(r"$q_z$", fontsize=26, rotation='horizontal')
    axes[2].set_xlabel(r"$\phi$", fontsize=26, rotation='horizontal')

    fig.text(0.855, 0.07, "[deg]", fontsize=16)
    fig.text(0.025, 0.49, "[km/s]", fontsize=16)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0., wspace=0.)
    fig.savefig(os.path.join(plot_path, "bootstrap.pdf"))

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
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-l", "--list", action="store_true", dest="list",
                        default=False, help="List all functions")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="Overwrite existing files.")
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

    func = getattr(sys.modules[__name__], args.__dict__.get("function"))
    func(**kwargs)

