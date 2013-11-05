# coding: utf-8

""" Select targets for Spitzer """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from datetime import datetime, timedelta

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.io.misc import fnpickle, fnunpickle
from scipy.interpolate import interp1d
from astropy.table import Table, Column, join
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.util import project_root
from streams.coordinates import distance_to_sgr_plane
from streams.io import add_sgr_coordinates
from streams.io.lm10 import particle_table

notes_path = os.path.join(project_root, "text", "notes",
                          "spitzer_target_selection")

def orphan():
    """ We'll select the high probability members from Branimir's
        Orphan sample.
    """
    filename = "branimir_orphan.txt"
    output_file = "orphan.txt"

    d = ascii.read(os.path.join(project_root, "data", "catalog", filename))
    high = d[d["membership_probability"] == "high"]
    high.keep_columns(["ID", "RA", "Dec", "magAvg", "period", "rhjd0"])
    high.rename_column("magAvg", "rMagAvg")

    ascii.write(high, \
        os.path.join(project_root, "data", "spitzer_targets", output_file))

def tbl_to_xyz(tbl):
    g = coord.ICRS(np.array(tbl["ra"])*u.deg, np.array(tbl["dec"])*u.deg,
                   distance=np.array(tbl["dist"])*u.kpc).galactic
    return g.x-8*u.kpc, g.y, g.z

def tbl_to_gc_dist(tbl):
    x,y,z = tbl_to_xyz(tbl)
    return np.sqrt(x**2 + y**2 + z**2)

def sgr_rv(d, lm10, Nbins=30, sigma_cut=3.):
    """ Select stars (d) that match in RV to the LM10 particles """

    # for each arm (trailing, leading)
    rv_funcs = {}
    median_rvs = {}
    Lambda_bins = {}
    rv_scatters = {}
    lmflag_idx = {}
    for lmflag in [-1,1]:
        wrap = lm10[lm10['Lmflag'] == lmflag]
        bins = np.linspace(wrap["Lambda"].min(), wrap["Lambda"].max(), Nbins)

        median_rv = []
        rv_scatter = []
        for ii in range(Nbins-1):
            binL = bins[ii]
            binR = bins[ii+1]
            idx = (wrap["Lambda"] > binL) & (wrap["Lambda"] < binR)
            median_rv.append(np.median(wrap["vgsr"][idx]))
            rv_scatter.append(np.std(wrap["vgsr"][idx]))

            # plt.clf()
            # plt.hist(wrap["vgsr"][idx])
            # plt.title("{0} - {1}".format(binL, binR))
            # plt.xlim(-250, 250)
            # plt.savefig(os.path.join(notes_path, "{0}_{1}.png".format(lmflag, int(binL))))

        Lambda_bins[lmflag] = (bins[1:]+bins[:-1])/2.
        median_rvs[lmflag] = np.array(median_rv)
        rv_scatters[lmflag] = np.array(rv_scatter)
        rv_func = interp1d(Lambda_bins[lmflag],
                           median_rvs[lmflag],
                           kind='cubic', bounds_error=False)

        _idx = np.zeros_like(d["Lambda"]).astype(bool)
        for ii in range(Nbins-1):
            lbin = bins[ii]
            rbin = bins[ii+1]

            ix = (d["Lambda"] >= lbin) | (d["Lambda"] < rbin)
            pred_rv = rv_func(d["Lambda"])
            # ix &= (d["Vgsr"] < (pred_rv+sigma_cut/sigma_cut*rv_scatters[lmflag][ii])) &\
            #       (d["Vgsr"] > (pred_rv-sigma_cut/sigma_cut*rv_scatters[lmflag][ii]))
            ix &= np.fabs(d["Vgsr"] - pred_rv) < rv_scatters[lmflag][ii]

            _idx |= ix

        lmflag_idx[lmflag] = _idx

    fig,axes = plt.subplots(1,2,figsize=(15,6))

    for ii,lmflag in enumerate([1,-1]):
        ax = axes[ii]
        wrap = lm10[lm10['Lmflag'] == lmflag]

        ax.plot(wrap["Lambda"], wrap["vgsr"],
                marker=',', linestyle='none', alpha=0.25)
        ax.plot(Lambda_bins[lmflag], median_rvs[lmflag], "k")
        ax.plot(Lambda_bins[lmflag],
                median_rvs[lmflag]+sigma_cut*rv_scatters[lmflag], c='g')
        ax.plot(Lambda_bins[lmflag],
                median_rvs[lmflag]-sigma_cut*rv_scatters[lmflag], c='g')

        selected_d = d[lmflag_idx[lmflag]]
        not_selected_d = d[~lmflag_idx[lmflag]]
        ax.plot(selected_d["Lambda"], selected_d["Vgsr"],
                marker='.', linestyle='none', alpha=0.75, c='#CA0020', ms=6)
        ax.plot(not_selected_d["Lambda"], not_selected_d["Vgsr"],
                marker='.', linestyle='none', alpha=0.75, c='#2B8CBE', ms=5)

        ax.set_xlim(0,360)
        ax.set_xlabel(r"$\Lambda$ [deg]")

        if ii == 0:
            ax.set_ylabel(r"$v_{\rm gsr}$ [km/s]")

        if lmflag == 1:
            ax.set_title("Leading", fontsize=20, fontweight='normal')
        elif lmflag == -1:
            ax.set_title("Trailing", fontsize=20, fontweight='normal')

    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "vgsr_selection.pdf"))

    return lmflag_idx

def sgr(overwrite=False):

    lm10_cache = os.path.join(project_root, "data", "spitzer_targets",
                              "lm10_cache.pickle")
    if os.path.exists(lm10_cache) and overwrite:
        os.remove(lm10_cache)

    if not os.path.exists(lm10_cache):
        # select particle data from the LM10 simulation
        lm10 = particle_table(N=0, expr="(Pcol>-1) & (Pcol<8) & "\
                                        "(abs(Lmflag)==1) & (dist<100)")
        fnpickle(np.array(lm10), lm10_cache)
    else:
        lm10 = Table(fnunpickle(lm10_cache))

    # read in the Catalina RR Lyrae data
    spatial_data = ascii.read(os.path.join(project_root,
                              "data/catalog/Catalina_all_RRLyr.txt"))
    velocity_data = ascii.read(os.path.join(project_root,
                               "data/catalog/Catalina_vgsr_RRLyr.txt"))
    catalina = join(spatial_data, velocity_data, join_type='outer', keys="ID")
    catalina.rename_column("RAdeg", "ra")
    catalina.rename_column("DEdeg", "dec")
    catalina.rename_column("dh", "dist")

    # add Sgr coordinates to the Catalina data
    catalina = add_sgr_coordinates(catalina)

    # add Galactocentric distance to the Catalina and LM10 data
    cat_gc_dist = tbl_to_gc_dist(catalina)
    lm10_gc_dist = tbl_to_gc_dist(lm10)
    catalina.add_column(Column(cat_gc_dist, name="gc_dist"))
    lm10.add_column(Column(lm10_gc_dist, name="gc_dist"))

    # 1) Select stars < 20 kpc from the orbital plane of Sgr
    sgr_catalina = catalina[np.fabs(catalina["Z_sgr"]) < 20.]
    x,y,z = tbl_to_xyz(sgr_catalina)

    # 2) Stars with D > 15 kpc from the Galactic center
    sgr_catalina = sgr_catalina[sgr_catalina["gc_dist"] > 15]
    sgr_catalina_rv = sgr_catalina[~sgr_catalina["Vgsr"].mask]
    print("{0} CSS RRLs have radial velocities.".format(len(sgr_catalina_rv)))

    # plot X-Z plane, data and lm10 particles
    fig,axes = plt.subplots(1,3,figsize=(15,6), sharex=True, sharey=True)
    x,y,z = tbl_to_xyz(catalina)
    axes[0].set_title("All RRL", fontsize=20)
    axes[0].plot(x, z, marker='.', alpha=0.2, linestyle='none')
    axes[0].set_xlabel("$X_{gc}$ kpc")
    axes[0].set_ylabel("$Z_{gc}$ kpc")

    x,y,z = tbl_to_xyz(sgr_catalina)
    axes[1].set_title(r"RRL $|Z-Z_{sgr}|$ $<$ $20$ kpc", fontsize=20)
    axes[1].plot(x, z, marker='.', alpha=0.2, linestyle='none')
    axes[2].plot(lm10["x"], lm10["z"], marker='.',
                 alpha=0.2, linestyle='none')
    axes[2].set_title("LM10", fontsize=20)
    axes[2].set_xlim(-60, 40)
    axes[2].set_ylim(-60, 60)

    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "catalina_all.pdf"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Select on radial velocities
    rv_selection_cache = os.path.join(project_root, "data",
                                      "spitzer_targets", "rv.pickle")
    if os.path.exists(rv_selection_cache) and overwrite:
        os.remove(rv_selection_cache)

    if not os.path.exists(rv_selection_cache):
        lmflag_rv_idx = sgr_rv(sgr_catalina_rv, lm10, Nbins=60, sigma_cut=3.)
        fnpickle(lmflag_rv_idx, rv_selection_cache)
    else:
        lmflag_rv_idx = fnunpickle(rv_selection_cache)

    # Make X-Z plot
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    for lmflag in [1,-1]:
        ix = lmflag_rv_idx[lmflag]
        x,y,z = tbl_to_xyz(sgr_catalina_rv[ix])
        ax.plot(x, z, marker='.', alpha=0.75,
                linestyle='none', ms=6, label="Lmflag={0}".format(lmflag))

    x,y,z = tbl_to_xyz(lm10)
    ax.plot(x, z, marker=',', alpha=0.2,
                linestyle='none')
    ax.legend(loc='lower right',\
              prop={'size':12})
    ax.set_title("RV-selected CSS RRLs", fontsize=20)
    ax.set_xlabel(r"$X_{\rm gc}$ kpc")
    ax.set_ylabel(r"$Z_{\rm gc}$ kpc")
    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "vgsr_selected_xz.pdf"))

    # Make Lambda-Beta plot
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    for lmflag in [1,-1]:
        ix = lmflag_rv_idx[lmflag]
        dd = sgr_catalina_rv[ix]
        ax.plot(dd["Lambda"], dd["Beta"], marker='.', alpha=0.75,
                linestyle='none', ms=6, label="Lmflag={0}".format(lmflag))

    ax.plot(lm10["Lambda"], lm10["Beta"], marker=',', alpha=0.2,
                linestyle='none')
    ax.legend(loc='lower right',\
              prop={'size':12})
    ax.set_title("RV-selected CSS RRLs", fontsize=20)
    ax.set_xlabel(r"$\Lambda$ [deg]")
    ax.set_ylabel(r"$B$ [deg]")
    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "vgsr_selected_LB.pdf"))

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        dest="overwrite", default=False)

    args = parser.parse_args()

    #orphan()
    sgr(args.overwrite)