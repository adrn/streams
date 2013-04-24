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

# Third-party
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy.stats import gaussian_kde
import astropy.coordinates as coord
from astropy.table import Table, Column
import astropy.units as u

from streams.data import *
from streams.coordinates import SgrCoordinates, OrphanCoordinates
from streams.data.gaia import rr_lyrae_photometric_distance

def discrete_cmap(N=8):
    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    cpool = ['#9E0142', '#D53E4F', '#F46D43', '#FDAE61', '#FEE08B',
             '#E6F598', '#ABDDA4', '#66C2A5', '#3288BD', '#5E4FA2']
    if N == 5:
        cmap3 = col.ListedColormap(cpool[::2], 'nice_spectral')
    else:
        cmap3 = col.ListedColormap(cpool[0:N], 'nice_spectral')
    cm.register_cmap(cmap=cmap3)

discrete_cmap(5)
css = read_catalina()
quest = read_quest()
asas = read_asas()
nsvs = read_nsvs()
sdss = read_stripe82()

css.keep_columns(["ra","dec","V"])
css.add_column(Column(1*np.ones(len(css)), name="survey"))
css = np.array(css["ra","dec","V","survey"])

quest.keep_columns(["ra","dec","V"])
quest.add_column(Column(2*np.ones(len(quest)), name="survey"))
quest = np.array(quest["ra","dec","V","survey"])

asas.keep_columns(["ra","dec","V"])
asas.add_column(Column(3*np.ones(len(asas)), name="survey"))
asas = np.array(asas["ra","dec","V","survey"])

nsvs.keep_columns(["ra","dec","V"])
nsvs.add_column(Column(4*np.ones(len(nsvs)), name="survey"))
nsvs = np.array(nsvs["ra","dec","V","survey"])

sdss.keep_columns(["ra","dec","V"])
sdss.add_column(Column(5*np.ones(len(sdss)), name="survey"))
sdss = np.array(sdss)

all_rrlyr = np.hstack((css, quest, asas, nsvs, sdss))
all_rrlyr = Table(all_rrlyr)

def sgr():    
    L,B = [], []
    for row in all_rrlyr:
        icrs = coord.ICRSCoordinates(row["ra"], row["dec"], unit=(u.degree, u.degree))
        sgr = icrs.transform_to(SgrCoordinates)
        #sgr = icrs.transform_to(OrphanCoordinates)
        L.append(sgr.Lambda.degrees)
        B.append(sgr.Beta.degrees)
    
    all_rrlyr.add_column(Column(L, name="Lambda_sgr"))
    all_rrlyr.add_column(Column(B, name="Beta_sgr"))
    
    lm10 = LM10Snapshot(expr="(Pcol > 0) & (dist < 100) & (abs(Lmflag) == 1)")
    
    xbins = np.linspace(0,360,100)
    ybins = np.linspace(0,100,25)
    H, x, y = np.histogram2d(lm10["lambda"], lm10["dist"], bins=(xbins,ybins))
    
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
    sgr()
    orphan()