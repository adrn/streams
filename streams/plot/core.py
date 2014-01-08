# coding: utf-8

""" General plotting utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle

# Third-party
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Ellipse
import astropy.units as u
from scipy.stats import gaussian_kde

__all__ = ["discrete_cmap", "bootstrap_scatter_plot", "sgr_kde", "plot_point_cov", "plot_cov_ellipse"]

def discrete_cmap(N=8):
    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    # #9E0142
    cpool = ['#D085A4', '#D53E4F', '#F46D43', '#FDAE61', '#FEE08B',
             '#E6F598', '#ABDDA4', '#66C2A5', '#3288BD', '#5E4FA2']
    if N == 5:
        cmap3 = col.ListedColormap(cpool[::2], 'nice_spectral')
    else:
        cmap3 = col.ListedColormap(cpool[0:N], 'nice_spectral')
    cm.register_cmap(cmap=cmap3)

def bootstrap_scatter_plot(d, subtitle="", axis_lims=None):
    """ """
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    fig,axes = plt.subplots(3, 3, figsize=(12,12))

    # Create dictionary of true parameter values, not quantities
    params = dict()
    for p in d.keys():
        params[p] = true_params[p]

        if p == 'phi':
            params[p] = params[p].value
        elif p == 'v_halo':
            params[p] = params[p].to(u.km/u.s).value

    # if no limits are provided, defined default axis limits
    axis_lims = dict(q1=(1.,2),
                     qz=(1.,2.),
                     phi=(np.pi/4, 3*np.pi/4),
                     v_halo=(100, 150))

    # Create latex labels for the parameters
    labels = dict(q1="$q_1$", qz="$q_z$",
                  phi="$\phi$ [rad]", v_halo="$v_{halo}$ [km/s]")

    # helper function for plotting single axes
    def plot_2params(ax, p1, p2):
        x = d[p1]
        y = d[p2]
        if p1 == 'v_halo':
            x = (d[p1]*u.kpc/u.Myr).to(u.km/u.s).value
        elif p2 == 'v_halo':
            y = (d[p2]*u.kpc/u.Myr).to(u.km/u.s).value

        ax.scatter(x, y, color='k', marker='.', alpha=0.5)
        ax.set_xlim(axis_lims[p1])
        ax.set_ylim(axis_lims[p2])
        ax.axvline(params[p1], linestyle="--", color="#555555")
        ax.axhline(params[p2], linestyle="--", color="#555555")

    q_ticks = np.arange(1.2,1.8,0.2)
    v_ticks = np.arange(105,155,10)
    phi_ticks = np.arange(1.0,2.2,0.2)
    plot_2params(axes[0,0], "q1", "qz")
    axes[0,0].set_ylabel(labels["qz"])
    axes[0,0].set_yticks(q_ticks)

    plot_2params(axes[1,0], "q1", "phi")
    axes[1,0].set_ylabel(labels["phi"])
    axes[1,0].set_yticks(phi_ticks)

    plot_2params(axes[2,0], "q1", "v_halo")
    axes[2,0].set_xlabel(labels["q1"])
    axes[2,0].set_ylabel(labels["v_halo"])
    axes[2,0].set_xticks(q_ticks)
    axes[2,0].set_yticks(v_ticks)

    plot_2params(axes[1,1], "qz", "phi")
    plot_2params(axes[2,1], "qz", "v_halo")
    axes[2,1].set_xlabel(labels["qz"])
    axes[2,1].set_xticks(q_ticks)

    plot_2params(axes[2,2], "phi", "v_halo")
    axes[2,2].set_xlabel(labels["phi"])
    axes[2,2].set_xticks(phi_ticks)

    axes[0,1].set_visible(False);axes[0,2].set_visible(False);axes[1,2].set_visible(False)
    axes[0,0].set_xticklabels([]);axes[1,0].set_xticklabels([]);axes[1,1].set_xticklabels([])
    axes[1,1].set_yticklabels([]);axes[2,1].set_yticklabels([]);axes[2,2].set_yticklabels([])

    for ax in fig.axes:
        if ax.get_xlabel() != "":
            ax.xaxis.label.set_size(24)
        if ax.get_ylabel() != "":
            ax.yaxis.label.set_size(24)

    fig.suptitle("Inferred halo parameters for 100 bootstrapped realizations", fontsize=28)
    fig.text(.5,.91, subtitle, fontsize=24, ha='center')
    fig.subplots_adjust(hspace=0.04, wspace=0.04)

    return fig

def sgr_kde(particle_data, extent={'x':(-85,85), 'y':(-85,85)}):
    """ TODO """

    # hack to get every 25th particle still bound to Sgr, otherwise this peak
    #   dominates the whole density field
    idx = (np.fabs(particle_data['Lmflag']) == 1) & (particle_data['dist'] < 70) \
            & (particle_data['Pcol'] > -1) & (particle_data['Pcol'] < 7)
    still_bound = np.zeros_like(idx)
    w, = np.where(particle_data['Pcol']==-1)
    still_bound[w[::50]] = 1.
    idx = idx | still_bound.astype(bool)

    X,Y = np.mgrid[extent['x'][0]:extent['x'][1]:400j,
                   extent['y'][0]:extent['y'][1]:400j]
    positions = np.vstack([X.ravel(),Y.ravel()])

    m1,m2 = particle_data[idx]['x'], particle_data[idx]['z']
    values = np.vstack([m1, m2])

    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape).T
    Z = np.flipud(Z)
    #Z = np.fliplr(Z)

    return Z

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip