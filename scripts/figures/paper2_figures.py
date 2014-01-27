# coding: utf-8

""" A script for making figures for our Streams Paper 1 """

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
matplotlib.use("agg")
import daft
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc_context, rcParams, cm
from matplotlib.patches import Rectangle, Ellipse
import scipy.optimize as so
from scipy.stats import norm

from streams.util import project_root
from streams.io.sgr import SgrSimulation
from streams.integrate import LeapfrogIntegrator
from streams.potential.lm10 import LawMajewski2010

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
#matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', edgecolor='#444444', labelsize=24,
              labelweight=400, linewidth=2.0)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='Source Sans Pro')
#matplotlib.rc('savefig', bbox='standard')

plot_path = "plots/paper2/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def graphical_model(**kwargs):

    filename = os.path.join(plot_path, "graphical_model.png")

    # Instantiate the PGM.
    pgm = daft.PGM([3.5, 2.5], origin=[0.3, 0.3])

    # Hierarchical parameters.
    pgm.add_node(daft.Node("sigma_x", r"$\Sigma_{\rm p}$,$x_{\rm p}$", 0.5, 2, fixed=True))
    pgm.add_node(daft.Node("beta", r"$\beta$", 1.5, 2))

    # Latent variable.
    pgm.add_node(daft.Node("w", r"$w_n$", 1, 1))

    # Data.
    pgm.add_node(daft.Node("x", r"$x_n$", 2, 1, observed=True))

    # Add in the edges.
    pgm.add_edge("sigma_x", "beta")
    pgm.add_edge("beta", "w")
    pgm.add_edge("w", "x")
    pgm.add_edge("beta", "x")

    # And a plate.
    pgm.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$n = 1, \cdots, N$",
        shift=-0.1))

    # Render and save.
    pgm.render()
    #pgm.figure.savefig("classic.pdf")
    pgm.figure.savefig(filename, dpi=150)

def gaussian(x, mu, sigma):
    return 1./np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5 * ((x-mu)/sigma)**2)

def rel_dist(**kwargs):

    nbins = 50
    fig,axes = plt.subplots(4,2,figsize=(8.5,11), sharex='col', sharey=True)
    fig2,axes2 = plt.subplots(1,2,figsize=(11,8.5))

    #_m = kwargs["mass"]
    for kk,_m in enumerate(["2.5e6","2.5e7","2.5e8","2.5e9"]):
        m = float(_m)
        sgr = SgrSimulation(_m)
        p = sgr.particles(N=5000, expr="tub!=0")
        p_bound = sgr.particles(N=0, expr="tub==0")
        v_disp = np.sqrt(np.var(p_bound["vx"]) + \
                         np.var(p_bound["vy"]) + \
                         np.var(p_bound["vz"])).value[0]
        s = sgr.satellite()

        lm10_potential = LawMajewski2010()

        X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
        V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
        integrator = LeapfrogIntegrator(lm10_potential._acceleration_at,
                                        np.array(X), np.array(V),
                                        args=(X.shape[0], np.zeros_like(X)))
        ts, rs, vs = integrator.run(t1=sgr.t1, t2=sgr.t2, dt=-1.)

        s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
        p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
        t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

        p_x = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
        s_x = np.array([s_orbit[jj,0] for jj in t_idx])

        diff = p_x-s_x
        rel_r = np.sqrt(np.sum(diff[:,:3]**2, axis=-1))
        rel_v = np.sqrt(np.sum(diff[:,3:]**2, axis=-1))

        r_tide = lm10_potential._tidal_radius(m, s_x)*1.6
        v_esc = lm10_potential._escape_velocity(m, r_tide)

        mu_r, sig_r = norm.fit(np.log(rel_r))
        print("effective disruption radius: {}".format(mu_r))
        print("\t\tspread: {}".format(sig_r))
        print("\t\tmean tidal radius: {}".format(np.log(r_tide.mean())))

        mu_v, sig_v = norm.fit(np.log(rel_v))
        print("effective disruption velocity: {}".format(mu_v))
        print("\t\tspread: {}".format(sig_v))
        print("\t\tvelocity dispersion: {}".format(np.log(v_disp)))
        print("\n")

        n,bins,patches = axes[kk,0].hist(np.log(rel_r), bins=nbins,
                                         alpha=0.4, normed=True)
        axes[kk,0].plot(bins, gaussian(bins, mu_r, sig_r), lw=2., alpha=0.75)
        axes[kk,0].axvline(np.log(r_tide.mean()), color='#ca0020', lw=2.,
                        alpha=0.75, label=r'$\ln \langle R_{\rm tide}\rangle$')
        axes[kk,0].set_yticks([])

        ylim = axes[kk,0].get_ylim()
        axes[kk,0].text(-2.75, ylim[1]-(ylim[1]-ylim[0])/5.,
                        r"{} $M_\odot$".format(_m),
                        fontsize=16)

        axes[kk,0].text(-2.75, ylim[1]-1.75*(ylim[1]-ylim[0])/5.,
                        r"$\sigma={:.2f}$".format(sig_r),
                        fontsize=14)

        ylim = axes[kk,1].get_ylim()
        axes[kk,1].text(-8.75, ylim[1]-1.75*(ylim[1]-ylim[0])/5.,
                        r"$\sigma={:.2f}$".format(sig_v),
                        fontsize=14)

        n,bins,patches = axes[kk,1].hist(np.log(rel_v), bins=nbins,
                                         alpha=0.4, normed=True)
        axes[kk,1].plot(bins, gaussian(bins, mu_v, sig_v), lw=2., alpha=0.75)
        axes[kk,1].axvline(np.log(v_disp), color='#2ca25f', lw=2.,
                        alpha=0.75, label=r'$\ln \langle \sigma_v\rangle$')
        axes[kk,1].set_yticks([])

        if kk == 0:
            axes[kk,0].set_title("relative distances")
            axes[kk,1].set_title("relative velocities")
            axes[kk,0].legend(fontsize=12)
            axes[kk,1].legend(fontsize=12)

        if kk == 3:
            axes[kk,0].set_xlabel(r"$\ln\vert r_i-r_{\rm sat} \vert$")
            axes[kk,1].set_xlabel(r"$\ln\vert v_i-v_{\rm sat} \vert$")

        n,bins,patches = axes2[0].hist((np.log(rel_r)-np.log(r_tide)), linewidth=1.,
                                       linestyle='solid', edgecolor='k', histtype='stepfilled',
                                       bins=nbins, alpha=0.25, normed=True)
        axes2[0].set_title("normalized distance distribution")
        axes2[0].set_yticks([])
        axes2[0].set_xlabel(r"$\ln\vert r_i-r_{\rm sat} \vert - \ln r_{\rm tide}$")

        n,bins,patches = axes2[1].hist((np.log(rel_v)-np.log(v_disp)), linewidth=1.,
                                       linestyle='solid', edgecolor='k', histtype='stepfilled',
                                       bins=nbins, alpha=0.25, normed=True)
        axes2[1].set_title("normalized velocity distribution")
        axes2[1].set_yticks([])
        axes2[1].set_xlabel(r"$\ln\vert v_i-v_{\rm sat} \vert - \ln \sigma_v$")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_path, "rel_dists.pdf"))

    fig2.tight_layout()
    axes2[0].set_xlim(-3,3)
    axes2[1].set_xlim(-3,3)
    fig2.savefig(os.path.join(plot_path, "norm_dists.pdf".format(_m)))

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

