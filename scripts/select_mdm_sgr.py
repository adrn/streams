# coding: utf-8

""" Select a sample of RR Lyrae associated spatially with Sagittarius 
    to observe at MDM 
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import astropy.table as at
from astropy.io import ascii, fits
import matplotlib
matplotlib.use("WxAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import leastsq
import ebf

from streams.data import read_catalina
from streams.data.gaia import rr_lyrae_photometric_distance
from streams.coordinates.sgr import distance_to_sgr_plane, SgrCoordinates

def galaxia():
    # Galaxia simulation data
    data = ebf.read("data/galaxia/galaxy_A.ebf", "/")
    glon = data["glon"]
    glon[glon > 180.] = glon[glon > 180.] - 360
    glat = data["glat"]
    
    # Compute the mean metallicity
    Fe_H = data['feh']
    
    # From Sandage 2006:
    # http://iopscience.iop.org/1538-3881/131/3/1750/pdf/204918.web.pdf
    B_minus_V_blue_edge = lambda Fe_H: 0.35 + 0.159*Fe_H + 0.05*Fe_H**2
    
    # Carretta & Gratton 1997 (or http://arxiv.org/pdf/astro-ph/0507464v2.pdf)
    true_M_V = lambda Fe_H: 0.23*Fe_H + 0.93
    
    B_V = data['ubv_b'] - data['ubv_v']
    M_V = data['ubv_v']
    
    # Instability strip is ~0.25 - blue edge
    FBE = np.mean(B_minus_V_blue_edge(Fe_H))
    
    idx = (B_V < 0.6) & (B_V > FBE) & \
          (M_V < true_M_V(Fe_H.max())) & (M_V > true_M_V(Fe_H.min())) & \
          (glon > -10.) & (glon < 10) & \
          (glat > 35) & (glat < 55)
          
    d = np.sqrt(data["px"]**2 + data["py"]**2 + data["pz"]**2)
    mu = 5.*np.log10(d) - 5
    m_V = mu + M_V
    
    n_galaxia, bins = np.histogram(d[idx], bins=25, density=False)
    n_galaxia = n_galaxia.astype(float) / 40. # sq. deg. -> # per sq. deg.

def data():
    # Now, select Catalina RR Lyrae 
    catalina = read_catalina()
    
    targets = dict()
    targets["Lambda"] = []
    targets["Beta"] = []
    targets["ra"] = []
    targets["dec"] = []
    targets["V"] = []
    targets["dist"] = []
    targets["zsun"] = []
    for star in catalina:
        dist = rr_lyrae_photometric_distance(star["V"], -1.5)
        dist_from_plane = distance_to_sgr_plane(star["ra"], star["dec"], dist)
        
        icrs = coord.ICRSCoordinates(star["ra"], star["dec"], unit=(u.degree, u.degree))
        sgr = icrs.transform_to(SgrCoordinates)
        
        # select Sgr associated stuff in northern sky
        if abs(dist_from_plane) < 10. \
            and sgr.Lambda.degrees > 180. and sgr.Lambda.degrees < 315 \
            and dist < 31*u.kpc:
            targets["ra"].append(star["ra"])
            targets["dec"].append(star["ra"])
            targets["V"].append(star["V"])
            targets["Lambda"].append(sgr.Lambda.degrees)
            targets["Beta"].append(sgr.Beta.degrees)
            targets["dist"].append(dist.value)
            targets["zsun"].append(dist.value * np.sin(sgr.Beta.radians))
    
    t = at.Table(targets)
    t = t['ra','dec','Lambda','Beta','V','dist','zsun']
    print("{0} stars selected".format(len(t)))
    
    t.write("data/catalog/Catalina_sgr_nearby_wrap.txt", format="ascii")
    
    return
    
    plt.figure(figsize=(8,8))
    plt.title("RR Lyrae from Catalina Sky Survey")
    plt.scatter(targets["Lambda"], 
                targets["zsun"],
                marker='.',
                color='k')
    plt.xlim(160,340)
    plt.ylim(-20,20)
    plt.xlabel("$\Lambda_\odot$ [deg]")
    plt.ylabel("$z_{\odot,sgr}$ [kpc]")
    plt.show()

if __name__ == "__main__":
    data()

