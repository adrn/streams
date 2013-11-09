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
from astropy.table import Table, Column, join, vstack
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
            ix &= np.fabs(d["Vgsr"] - pred_rv) < np.sqrt(10**2 + rv_scatters[lmflag][ii]**2)

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

def sgr_dist(d, lm10, Nbins=30, sigma_cut=3.):
    """ Select stars (d) that match in distance to the LM10 particles """

    # for each arm (trailing, leading)
    lmflag_idx = {}
    median_dists = {}
    Lambda_bins = {}
    dist_scatters = {}
    for lmflag in [-1,1]:
        wrap = lm10[lm10['Lmflag'] == lmflag]
        dist = wrap["dist"]
        bins = np.linspace(wrap["Lambda"].min(), wrap["Lambda"].max(), Nbins)

        median_dist = []
        dist_scatter = []
        for ii,binL in enumerate(bins):
            if ii == Nbins-1: break

            binR = bins[ii+1]
            idx = (wrap["Lambda"] > binL) & (wrap["Lambda"] < binR)

            m = np.median(dist[idx])
            s = np.std(dist[idx])
            if 300 > binL > 225 and lmflag == 1:
                m -= 5.
                s *= 1.
            elif 90 > binL > 40 and lmflag == 1:
                s /= 2.

            dist_scatter.append(s)
            median_dist.append(m)

        Lambda_bins[lmflag] = (bins[1:]+bins[:-1])/2.
        median_dists[lmflag] = np.array(median_dist)
        dist_scatters[lmflag] = np.array(dist_scatter)
        dist_func = interp1d(Lambda_bins[lmflag],
                             median_dists[lmflag],
                             kind='cubic', bounds_error=False)
        scatter_func = interp1d(Lambda_bins[lmflag],
                                dist_scatters[lmflag],
                                kind='cubic', bounds_error=False)

        _idx = np.zeros_like(d["Lambda"]).astype(bool)
        for ii in range(Nbins-1):
            lbin = bins[ii]
            rbin = bins[ii+1]

            ix = (d["Lambda"] >= lbin) | (d["Lambda"] < rbin)
            pred_dist = dist_func(d["Lambda"])
            pred_scat = scatter_func(d["Lambda"])

            ix &= np.fabs(d["dist"] - pred_dist) < sigma_cut*pred_scat
            #dist_scatters[lmflag][ii]

            _idx |= ix

        lmflag_idx[lmflag] = _idx

    fig,axes = plt.subplots(1,2,figsize=(15,6))

    for ii,lmflag in enumerate([1,-1]):
        ax = axes[ii]
        wrap = lm10[lm10['Lmflag'] == lmflag]

        ax.plot(wrap["Lambda"], wrap["dist"],
                marker=',', linestyle='none', alpha=0.25)
        ax.plot(Lambda_bins[lmflag], median_dists[lmflag], "k")
        ax.plot(Lambda_bins[lmflag],
                median_dists[lmflag]+sigma_cut*dist_scatters[lmflag], c='g')
        ax.plot(Lambda_bins[lmflag],
                median_dists[lmflag]-sigma_cut*dist_scatters[lmflag], c='g')

        selected_d = d[lmflag_idx[lmflag]]
        not_selected_d = d[~lmflag_idx[lmflag]]
        ax.plot(selected_d["Lambda"], selected_d["dist"],
                marker='.', linestyle='none', alpha=0.75, c='#CA0020', ms=6)
        ax.plot(not_selected_d["Lambda"], not_selected_d["dist"],
                marker='.', linestyle='none', alpha=0.75, c='#2B8CBE', ms=5)

        ax.set_xlim(0,360)
        ax.set_xlabel(r"$\Lambda$ [deg]")

        if ii == 0:
            ax.set_ylabel(r"$d_{\odot}$ [kpc]")

        if lmflag == 1:
            ax.set_title("Leading", fontsize=20, fontweight='normal')
        elif lmflag == -1:
            ax.set_title("Trailing", fontsize=20, fontweight='normal')

    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "dist_selection.pdf"))

    return lmflag_idx

def select_only(ixx, N):
    w, = np.where(ixx)
    ix = np.zeros_like(ixx).astype(bool)
    np.random.shuffle(w)
    try:
        w = w[:N]
        ix[w] = True
    except:
        ix = np.ones_like(ixx).astype(bool)
    return ix

def integration_time(d):
    d = np.array(d)
    f = d/10. #kpc
    return (3.*f**2*12*u.minute).to(u.hour)

def sgr(overwrite=False, seed=42):

    np.random.seed(seed)

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

    # plot Lambda-dist plane, data and lm10 particles
    fig,ax = plt.subplots(1,1,figsize=(6,6),
                          subplot_kw=dict(projection="polar"))

    ax.set_theta_direction(-1)
    ax.plot(np.radians(lm10["Lambda"]), lm10["dist"],
                       marker='.', alpha=0.2, linestyle='none')
    ax.plot(np.radians(sgr_catalina["Lambda"]), sgr_catalina["dist"],
                       marker='.', alpha=0.75, linestyle='none', ms=6)

    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "rrl_over_lm10.pdf"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Select on radial velocities
    rv_selection_cache = os.path.join(project_root, "data",
                                      "spitzer_targets", "rv.pickle")
    if os.path.exists(rv_selection_cache) and overwrite:
        os.remove(rv_selection_cache)

    if not os.path.exists(rv_selection_cache):
        lmflag_rv_idx = sgr_rv(sgr_catalina_rv, lm10, Nbins=50, sigma_cut=3.)
        fnpickle(lmflag_rv_idx, rv_selection_cache)
    else:
        lmflag_rv_idx = fnunpickle(rv_selection_cache)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Select on distance
    _selection_cache = os.path.join(project_root, "data",
                                    "spitzer_targets", "dist.pickle")
    if os.path.exists(_selection_cache) and overwrite:
        os.remove(_selection_cache)

    if not os.path.exists(_selection_cache):
        lmflag_dist_idx = sgr_dist(sgr_catalina_rv, lm10,
                                 Nbins=40, sigma_cut=2.5)
        fnpickle(lmflag_dist_idx, _selection_cache)
    else:
        lmflag_dist_idx = fnunpickle(_selection_cache)

    ################################################################

    # Make X-Z plot
    fig,ax = plt.subplots(1,1,figsize=(6,6))

    x,y,z = tbl_to_xyz(lm10)
    ax.plot(x, z, marker=',', alpha=0.2,
                linestyle='none')

    for lmflag in [1,-1]:
        ix = lmflag_dist_idx[lmflag] & lmflag_rv_idx[lmflag]
        x,y,z = tbl_to_xyz(sgr_catalina_rv[ix])
        ax.plot(x, z, marker='.', alpha=0.75,
                linestyle='none', ms=6, label="Lmflag={0}".format(lmflag))

    ax.legend(loc='lower right',\
              prop={'size':12})
    ax.set_title("RV-selected CSS RRLs", fontsize=20)
    ax.set_xlabel(r"$X_{\rm gc}$ kpc")
    ax.set_ylabel(r"$Z_{\rm gc}$ kpc")
    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "selected_xz.pdf"))

    ##############################################################
    # Finalize two samples:
    #   - 1 with 10 stars in the nearby trailing wrap
    #   - 1 without these stars

    L = sgr_catalina_rv["Lambda"]
    B = sgr_catalina_rv["Beta"]
    D = sgr_catalina_rv["dist"]
    X,Y = D*np.cos(np.radians(L)),D*np.sin(np.radians(L))

    lead_ix = lmflag_dist_idx[1] & lmflag_rv_idx[1]
    trail_ix = lmflag_dist_idx[-1] & lmflag_rv_idx[-1]
    trail_ix[L < 180] &= B[L < 180] < 5

    trail_ix &= np.logical_not((L > 50) & (L < 100) & (sgr_catalina_rv["dist"] < 40))

    # stars selected w/ distance and RV
    trail_ix_without = trail_ix & (L < 180)
    trail_ix_with = trail_ix & ( ((L > 230) & (L < 315)) | (L < 180) )

    # deselect stars possibly associated with the bifurcation
    no_bif = (L > 180) & (L < 360) & (B > -40) & (B < 0)
    no_bif |= ((L <= 180) & (B < 0) & (L > 80))

    # draw a box around some possible bifurcation members
    bifurcation_box  = (L > 210) & (L < 225) & (B > 0) & (B < 15) # OR
    print(sum(bifurcation_box), "bifurcation stars")
    print(sum(lead_ix), "leading arm stars")
    print(sum(trail_ix_with), "trailing arm stars (with nearby)")
    print(sum(trail_ix_without), "trailing arm stars (without nearby)")

    # only select 5 bifurcation particles
    ix_with = (lead_ix & (no_bif | bifurcation_box)) | trail_ix_with
    ix_bif = select_only(ix_with & bifurcation_box, 8)

    # select 3 clumps in the leading arm
    Nclump = 11
    ix_lead_clumps = np.zeros_like(lead_ix).astype(bool)
    for clump in [(215,20), (260,40)]:
        l,d = clump
        x,y = d*np.cos(np.radians(l)),d*np.sin(np.radians(l))

        # find N stars closest to the clump
        clump_dist = np.sqrt((X-x)**2 + (Y-y)**2)
        xxx = np.sort(clump_dist[lead_ix])[Nclump]

        this_ix = lead_ix & (clump_dist <= xxx)
        print("lead",sum(this_ix))
        ix_lead_clumps |= this_ix #select_only(this_ix, 10)
    ix_lead_clumps &= lead_ix

    # all southern leading
    lll = ((L > 80) & (L < 180) & lead_ix)
    print("southern leading", sum(lll))
    ix_lead_clumps |= lll

    # select all trailing stuff in south > 90
    # ix_trail_clumps_with = np.zeros_like(trail_ix).astype(bool)
    # ix_trail_clumps_without = np.zeros_like(trail_ix).astype(bool)
    # for clump in [(45,75), (75,100), (100, 135), (255, 300)]:
    #     this_ix_with = trail_ix_with & (L > clump[0]) & (L < clump[1])
    #     this_ix_without = trail_ix_without & (L > clump[0]) & (L < clump[1])

    #     ix_trail_clumps_with |= select_only(this_ix_with, 10)
    #     ix_trail_clumps_without |= select_only(this_ix_without, 10)
    ix_trail_clumps_without = trail_ix_without

    ix_trail_clumps_with = np.zeros_like(trail_ix).astype(bool)
    for clump in [(280,22)]:
        l,d = clump
        x,y = d*np.cos(np.radians(l)),d*np.sin(np.radians(l))

        # find N stars closest to the clump
        clump_dist = np.sqrt((X-x)**2 + (Y-y)**2)
        xxx = np.sort(clump_dist[trail_ix])[10]

        this_ix = trail_ix_with & (clump_dist < xxx)
        ix_trail_clumps_with |= this_ix #select_only(this_ix, 10)
    ix_trail_clumps_with &= trail_ix_with

    # all trailing southern
    ttt = (L > 90) & (L < 180) & (trail_ix_with)
    print("southern trailing", sum(ttt))
    ix_trail_clumps_with |= ttt

    i1 = integration_time(sgr_catalina_rv[ix_bif]["dist"])
    i2 = integration_time(sgr_catalina_rv[ix_lead_clumps]["dist"])
    i3_with = integration_time(sgr_catalina_rv[ix_trail_clumps_with]["dist"])
    i3_without = integration_time(sgr_catalina_rv[ix_trail_clumps_without]["dist"])

    print(len(sgr_catalina_rv[ix_bif]))
    print(len(sgr_catalina_rv[ix_lead_clumps]))
    print(len(sgr_catalina_rv[ix_trail_clumps_with]))
    targets_with = vstack((sgr_catalina_rv[ix_bif],
                           sgr_catalina_rv[ix_lead_clumps],
                           sgr_catalina_rv[ix_trail_clumps_with]))
    targets_without = vstack((sgr_catalina_rv[ix_bif],
                              sgr_catalina_rv[ix_lead_clumps],
                              sgr_catalina_rv[ix_trail_clumps_without]))

    print()
    print("bifurcation", np.sum(i1))
    print("leading",np.sum(i2))
    print("trailing",np.sum(i3_with))
    print("Total:",np.sum(integration_time(targets_with["dist"])))

    output_file = "sgr.txt"
    output = targets_with.copy()
    output.rename_column("Eta", "hjd0")
    output.rename_column("<Vmag>", "VMagAvg")
    output.keep_columns(["ID", "ra", "dec", "VMagAvg", "Period", "hjd0"])
    ascii.write(output,
                os.path.join(project_root, "data", "spitzer_targets", output_file), Writer=ascii.Basic)

    # =========================
    # =========================
    # with nearby trailing wrap
    ix_with = (lead_ix & (no_bif | bifurcation_box)) | trail_ix_with

    # ----------------------------------
    # Make Lambda-D plot WITH nearby trailing
    fig,ax = plt.subplots(1,1,figsize=(6,6),
                          subplot_kw=dict(projection="polar"))
    ax.set_theta_direction(-1)

    ax.plot(np.radians(lm10["Lambda"]), lm10["dist"],
            marker=',', alpha=0.2, linestyle='none')

    d = sgr_catalina_rv[ix_with]
    ax.plot(np.radians(d["Lambda"]), d["dist"], marker='.',
            alpha=0.75, linestyle='none', ms=6, c="#CA0020")

    d = targets_with
    ax.plot(np.radians(d["Lambda"]), d["dist"], marker='.',
            alpha=1., linestyle='none', ms=6, c="#31A354")

    ax.set_ylim(0,65)
    #ax.set_title("with nearby trailing", fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "with_near_trailing_xz.pdf"))

    # ---------------------
    # Make Lambda-Beta plot
    fig,ax = plt.subplots(1,1,figsize=(12,5))

    ax.plot(lm10["Lambda"], lm10["Beta"], marker=',', alpha=0.2,
                linestyle='none')

    dd = sgr_catalina_rv[ix_with]
    ax.plot(dd["Lambda"], dd["Beta"], marker='.', alpha=0.75,
            linestyle='none', ms=6, c="#CA0020")
    ax.plot(targets_with["Lambda"], targets_with["Beta"], marker='.',
            alpha=0.75, linestyle='none', ms=6, c="#31A354")

    ax.set_title("RV-selected CSS RRLs", fontsize=20)
    ax.set_xlabel(r"$\Lambda$ [deg]")
    ax.set_ylabel(r"$B$ [deg]")
    ax.set_xlim(360, 0)
    ax.set_ylim(-45, 45)
    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "with_near_trailing_LB.pdf"))

    # ===================
    # ===================
    # WITHOUT nearby wrap
    ix_without = (lead_ix & (no_bif | bifurcation_box)) | trail_ix_without

    # ----------------------------------
    # Make X-Z plot WITHOUT nearby trailing
    fig,ax = plt.subplots(1,1,figsize=(6,6),
                          subplot_kw=dict(projection="polar"))
    ax.set_theta_direction(-1)

    ax.plot(np.radians(lm10["Lambda"]), lm10["dist"],
            marker=',', alpha=0.2, linestyle='none')

    d = sgr_catalina_rv[ix_without]
    ax.plot(np.radians(d["Lambda"]), d["dist"], marker='.',
            alpha=0.75, linestyle='none', ms=6, c="#CA0020")

    d = targets_without
    ax.plot(np.radians(d["Lambda"]), d["dist"], marker='.',
            alpha=1., linestyle='none', ms=6, c="#31A354")

    ax.set_ylim(0,65)
    ax.set_title("without nearby trailing", fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "without_near_trailing_xz.pdf"))

    # ---------------------
    # Make Lambda-Beta plot
    fig,ax = plt.subplots(1,1,figsize=(12,5))

    ax.plot(lm10["Lambda"], lm10["Beta"], marker=',', alpha=0.2,
                linestyle='none')

    dd = sgr_catalina_rv[ix_without]
    dd_bif = sgr_catalina_rv[ix_without & bifurcation_box]
    ax.plot(dd["Lambda"], dd["Beta"], marker='.', alpha=0.75,
            linestyle='none', ms=6)
    ax.plot(dd_bif["Lambda"], dd_bif["Beta"], marker='.', alpha=0.75,
            linestyle='none', ms=6, c="#31A354", label="bifurcation")

    ax.set_title("RV-selected CSS RRLs", fontsize=20)
    ax.set_xlabel(r"$\Lambda$ [deg]")
    ax.set_ylabel(r"$B$ [deg]")
    ax.set_xlim(360, 0)
    ax.set_ylim(-45, 45)
    fig.tight_layout()
    fig.savefig(os.path.join(notes_path, "without_near_trailing_LB.pdf"))

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        dest="overwrite", default=False)

    args = parser.parse_args()

    #orphan()
    sgr(args.overwrite)