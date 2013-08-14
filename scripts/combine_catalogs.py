# coding: utf-8

""" Combine RR Lyrae data from Stripe 82, QUEST, NSVS, LINEAR, Catalina, ASAS

    Stripe 82 : comes with SDSS ugriz
    QUEST : already V-band, woot. -> MEAN V-band magnitude
    NSVS : already V-band! -> Also MEAN V-band magnitude
    LINEAR : LINEAR camera does not have a filter, but for RR Lyrae stars, 
             the response is basically the same as SDSS r. So, for RR Lyrae 
             stars, the LINEAR magnitudes are similar to SDSS r magnitudes 
             within 0.03 mag (rms). TODO: How to get V-band?
    Catalina : already V-band!
    ASAS : already V-band!
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from astropy.io import ascii
import astropy.coordinates as coord
from astropy.table import Table, Column
import astropy.units as u

from streams.data import *
from streams.coordinates import SgrCoordinates, OrphanCoordinates
from streams.data.gaia import rr_lyrae_photometric_distance
from streams.plot import discrete_cmap

# Create logger
logger = logging.getLogger(__name__)

discrete_cmap(5)
    
# Make sure plot dump directory exists:
plot_path = os.path.join("plots", "rr_lyr_surveys")
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

all_surveys_filename = os.path.join('data','catalog','all_surveys_rrlyr.txt ')
surveys = ["catalina", "quest", "asas", "nsvs", "stripe82"]

# TODO: Add a distance function for LINEAR, and add to here.

def compile_surveys():
    """ Compile RR Lyrae data from all surveys into one txt file """
    
    for ii,survey in enumerate(surveys):
        reader = globals()["read_{0}".format(survey)]
        data = reader()
        data.keep_columns(["ra","dec","V"])
        data.add_column(Column(ii*np.ones(len(data)), name="survey_id"))
        logger.info("Survey {0} = {1}".format(ii, survey))
        
        try:
            all_data = np.hstack((all_data, np.array(data["ra","dec","V","survey_id"])))
        except NameError:
            all_data = np.array(data["ra","dec","V","survey_id"])
    
    all_rrlyr = Table(all_data)
    
    L_sgr,B_sgr = [], []
    L_orp,B_orp = [], []
    for star in all_rrlyr:
        icrs = coord.ICRSCoordinates(star["ra"], star["dec"], unit=(u.degree, u.degree))
        
        sgr = icrs.transform_to(SgrCoordinates)
        L_sgr.append(sgr.Lambda.degrees)
        B_sgr.append(sgr.Beta.degrees)
        
        orp = icrs.transform_to(OrphanCoordinates)
        L_orp.append(orp.Lambda.degrees)
        B_orp.append(orp.Beta.degrees)
    
    all_rrlyr.add_column(Column(L_sgr, name="Lambda_sgr"))
    all_rrlyr.add_column(Column(B_sgr, name="Beta_sgr"))
    all_rrlyr.add_column(Column(L_orp, name="Lambda_orp"))
    all_rrlyr.add_column(Column(B_orp, name="Beta_orp"))
    
    all_rrlyr.write(all_surveys_filename, format="ascii")

def all_sky_plot(all_rrlyr):
    
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, projection="hammer")
    
    # cheat and get RA into the range -180,180 but make it look like 360,0
    ra = -(all_rrlyr["ra"]-180)
    ra = (ra*u.degree).to(u.radian).value
    dec = (all_rrlyr["dec"]*u.degree).to(u.radian).value
    s = ax.scatter(ra, 
                   dec, 
                   c=all_rrlyr["survey_id"], 
                   cmap=cm.get_cmap('nice_spectral'), 
                   edgecolor="none",
                   s=8,
                   alpha=0.75)
    ax.set_axis_bgcolor("#666666")
    ax.set_xticklabels([])
    cb = fig.colorbar(s, ax=ax)
    
    cbar_labels = []
    for survey in surveys:
        cbar_labels.append("")
        cbar_labels.append(survey)
    cb.ax.set_yticklabels(cbar_labels + [""])
    fig.savefig(os.path.join(plot_path, "all_sky_rr_lyr.png"))

def all_sky_sgr(all_rrlyr):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, projection="hammer")
    
    selected_rrlyr = all_rrlyr[np.fabs(all_rrlyr["Beta_sgr"]) < 10.]
    
    # cheat and get RA into the range -180,180 but make it look like 360,0
    ra = -(selected_rrlyr["ra"]-180)
    ra = (ra*u.degree).to(u.radian).value
    dec = (selected_rrlyr["dec"]*u.degree).to(u.radian).value
    s = ax.scatter(ra, 
                   dec, 
                   c=selected_rrlyr["survey_id"], 
                   cmap=cm.get_cmap('nice_spectral'), 
                   edgecolor="none",
                   s=8,
                   alpha=0.75)
    ax.set_axis_bgcolor("#666666")
    ax.set_xticklabels([])
    cb = fig.colorbar(s, ax=ax)
    
    cbar_labels = []
    for survey in surveys:
        cbar_labels.append("")
        cbar_labels.append(survey)
    cb.ax.set_yticklabels(cbar_labels + [""])
    fig.suptitle(r"RR Lyrae  $|B_{sgr}| < 10$ deg")
    fig.savefig(os.path.join(plot_path, "all_sky_sgr_rr_lyr.png"))

def all_sky_orp(all_rrlyr):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, projection="hammer")
    
    selected_rrlyr = all_rrlyr[np.fabs(all_rrlyr["Beta_orp"]) < 1.5]
    
    # cheat and get RA into the range -180,180 but make it look like 360,0
    ra = -(selected_rrlyr["ra"]-180)
    ra = (ra*u.degree).to(u.radian).value
    dec = (selected_rrlyr["dec"]*u.degree).to(u.radian).value
    s = ax.scatter(ra, 
                   dec, 
                   c=selected_rrlyr["survey_id"], 
                   cmap=cm.get_cmap('nice_spectral'), 
                   edgecolor="none",
                   s=8,
                   alpha=0.75)
    ax.set_axis_bgcolor("#666666")
    ax.set_xticklabels([])
    cb = fig.colorbar(s, ax=ax)
    
    cbar_labels = []
    for survey in surveys:
        cbar_labels.append("")
        cbar_labels.append(survey)
    cb.ax.set_yticklabels(cbar_labels + [""])
    fig.suptitle(r"RR Lyrae  $|B_{orp}| < 1.5$ deg")
    fig.savefig(os.path.join(plot_path, "all_sky_orphan_rr_lyr.png"))
    
    
def magic_plot(ax, x, y, c=None, s=None):
    s = ax.scatter(x, y, c=c, 
                   cmap=cm.get_cmap('nice_spectral'), 
                   edgecolor="none",
                   s=s,
                   alpha=0.5)
    ax.set_axis_bgcolor("#555555")
    if len(c) > 1:
        cb = ax.figure.colorbar(s, ax=ax, fraction=0.085)
        
        cbar_labels = []
        for survey in surveys:
            cbar_labels.append("")
            cbar_labels.append(survey)
        cb.ax.set_yticklabels(cbar_labels + [""])
    
    return ax
    
def sgr_plane_dist(all_rrlyr):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="polar")
    
    selected_rrlyr = all_rrlyr[np.fabs(all_rrlyr["Beta_sgr"]) < 10.]
    
    lam = (selected_rrlyr["Lambda_sgr"]*u.degree).to(u.radian).value
    dist = rr_lyrae_photometric_distance(selected_rrlyr["V"], -1.)
    ax = magic_plot(ax, lam, dist, c=selected_rrlyr["survey_id"], s=(np.array(dist)/10.)**1.5+4)
    ax.set_ylim(0., 80.)
    ax.set_xlabel(r"$\Lambda_{sgr}$ [deg]")
    fig.suptitle(r"RR Lyrae  $|B_{sgr}| < 10.$ deg")
    fig.savefig(os.path.join(plot_path, "sgr_Lambda_dist_polar.png"))
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="polar")
    lm10 = LM10Snapshot(expr="(Pcol > 0) & (dist < 100) & (abs(Lmflag) == 1) & (Beta < 10.)")
    lam = (lm10["Lambda"]*u.degree).to(u.radian).value
    dist = lm10["dist"]
    ax = magic_plot(ax, lam, dist, c="w", s=(np.array(dist)/10.)**1.5+4)
    ax.set_ylim(0., 80.)
    ax.set_xlabel(r"$\Lambda_{sgr}$ [deg]")
    fig.suptitle(r"Particles from Law & Majewski 2010")
    fig.savefig(os.path.join(plot_path, "lm10_Lambda_dist_polar.png"))
    
    return
    
    lm10 = LM10Snapshot(expr="(Pcol > 0) & (dist < 100) & (abs(Lmflag) == 1)")
    
    xbins = np.linspace(0,360,100)
    ybins = np.linspace(0,100,25)
    H, x, y = np.histogram2d(lm10["Lambda"], lm10["dist"], bins=(xbins,ybins))
    
    plt.figure(figsize=(12,8))
    plt.imshow(H.T, interpolation="nearest", extent=[0,360,100,0], 
               cmap=cm.gist_earth_r, aspect=3)
    plt.ylim(plt.ylim()[::-1])
    xlim = plt.xlim()
    ylim = plt.ylim()[::-1]
    plt.xlabel(r"$\Lambda_{sgr}$ [deg]")
    plt.ylabel(r"$D$ [kpc]")
    plt.tight_layout()
    plt.savefig("plots/sgr/lm10_density.png")
    
    # Now, particles in Sgr plane:
    sgr_idx = np.fabs(all_rrlyr["Beta_sgr"]) < 10.
    sgr_rrlyr = all_rrlyr[sgr_idx]
    dist = rr_lyrae_photometric_distance(sgr_rrlyr["V"], -1.5)
    
    plt.figure(figsize=(12,8))
    plt.scatter(sgr_rrlyr["Lambda_sgr"], rr_lyrae_photometric_distance(sgr_rrlyr["V"], -1.5), 
                edgecolor="none", c=sgr_rrlyr["survey"], s=8, marker="o", 
                cmap=cm.get_cmap('nice_spectral'), alpha=1.)
    plt.xlim(xlim)
    plt.ylim(ylim[::-1])
    plt.xlabel(r"$\Lambda_{sgr}$ [deg]")
    plt.ylabel(r"$D$ [kpc]")
    plt.savefig("plots/rrlyr_particles_sgr.png")
    
    H, x, y = np.histogram2d(sgr_rrlyr["Lambda_sgr"], dist, bins=(xbins,ybins))
    plt.figure(figsize=(12,8))
    plt.imshow(H.T, interpolation="nearest", extent=[0,360,100,0], 
               cmap=cm.gist_earth_r, aspect=3)
    plt.xlim(xlim)
    plt.ylim(ylim[::-1])
    plt.xlabel(r"$\Lambda_{sgr}$ [deg]")
    plt.ylabel(r"$D$ [kpc]")
    plt.tight_layout()
    plt.savefig("plots/rrlyr_density_sgr.png")

def orphan():    
    L,B = [], []
    for row in all_rrlyr:
        icrs = coord.ICRSCoordinates(row["ra"], row["dec"], unit=(u.degree, u.degree))
        sgr = icrs.transform_to(OrphanCoordinates)
        L.append(sgr.Lambda.degrees)
        B.append(sgr.Beta.degrees)
    
    all_rrlyr.add_column(Column(L, name="Lambda_orp"))
    all_rrlyr.add_column(Column(B, name="Beta_orp"))
    
    xbins = np.linspace(-180,180,100)
    ybins = np.linspace(0,100,25)
    xlim = (-180,180)
    ylim = (100,0)
    
    # From Newberg et al.
    l_orp = [-30,-20,-9,-1,8,18.4,36]
    d_orp = [46.8,40.7,32.4,29.5,24.5,21.4,18.6]
    dd_orp = [4.5,1.9,1.5,1.4,1.2,1.0,0.9]
    
    # Now, particles in orphan plane:
    sgr_idx = np.fabs(all_rrlyr["Beta_orp"]) < 2.
    sgr_rrlyr = all_rrlyr[sgr_idx]
    dist = rr_lyrae_photometric_distance(sgr_rrlyr["V"], -1.5)
    
    plt.figure(figsize=(12,8))
    plt.scatter(sgr_rrlyr["Lambda_orp"], rr_lyrae_photometric_distance(sgr_rrlyr["V"], -1.5), 
                edgecolor="none", c=sgr_rrlyr["survey"], s=8, marker="o", 
                cmap=cm.get_cmap('nice_spectral'), alpha=1.)
    plt.errorbar(l_orp, d_orp, yerr=dd_orp, color="r", alpha=0.6)
    plt.xlim(xlim)
    plt.ylim(ylim[::-1])
    plt.xlabel(r"$\Lambda_{orphan}$ [deg]")
    plt.ylabel(r"$D$ [kpc]")
    plt.savefig("plots/rrlyr_particles_orphan.png")
    
    H, x, y = np.histogram2d(sgr_rrlyr["Lambda_orp"], dist, bins=(xbins,ybins))
    plt.figure(figsize=(12,8))
    plt.imshow(H.T, interpolation="nearest", extent=[-180,180,100,0], 
               cmap=cm.gist_earth_r, aspect=3)
    plt.errorbar(l_orp, d_orp, yerr=dd_orp, color="r", alpha=0.6)
    plt.xlim(xlim)
    plt.ylim(ylim[::-1])
    plt.xlabel(r"$\Lambda_{orphan}$ [deg]")
    plt.ylabel(r"$D$ [kpc]")
    plt.tight_layout()
    plt.savefig("plots/rrlyr_density_orphan.png")

if __name__ == "__main__":
    
    if not os.path.exists(all_surveys_filename):
        compile_surveys()
    
    all_rrlyr = ascii.read(all_surveys_filename)
    
    #all_sky_plot(all_rrlyr)
    #all_sky_sgr(all_rrlyr)
    #all_sky_orp(all_rrlyr)
    sgr_plane_dist(all_rrlyr)
