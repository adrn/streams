# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from astropy.io import ascii
from astropy.coordinates import Angle
import astropy.units as u

from streams.data.sgr import LM10

def M_V(fe_h):
    """ Equation from: http://iopscience.iop.org/1538-3881/142/5/163/pdf/1538-3881_142_5_163.pdf """
    return 0.214 * (fe_h + 1.5) + 0.44

def sgr_kde():
    lm10 = LM10().data
    lm10 = lm10[lm10["Pcol"] > -1]
    lm10 = lm10[lm10["dist"] < 100]
    
    values = np.vstack([lm10["xsun"], lm10["ysun"], lm10["zsun"]])
    kernel = gaussian_kde(values)
    
    X, Y = np.mgrid[-100:100:200j, -100:100:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    
    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111)
    ax.imshow(np.rot90(Z), cmap=cm.gist_earth_r, extent=[-100, 100, 100, -100])
    fig.savefig("plots/sgr_kde.png")
    
def triand_3d():
    """ Metallicity : [Fe/H] ~ -1.2 
        
        M_V = 0.214 * ([Fe/H] + 1.5) + 0.44
    """
    
    triand = ascii.read("data/catalog/TriAnd_RRLyr.txt", 
                        include_names=["ra","dec","magAvg"])
    
    mu = triand["magAvg"] - M_V(-1.2)
    d = 10**(mu/5. + 1) * u.pc
    
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")
    ax = fig.add_subplot(111)
    
    from astropy.coordinates import ICRSCoordinates
    l,b = [],[]
    for line in triand:
        icrs = ICRSCoordinates(line["ra"], line["dec"], unit=(u.degree, u.degree))
        l.append(icrs.galactic.l.degrees)
        b.append(icrs.galactic.b.degrees)
    
    #ra = [Angle(r, unit=u.degree, bounds=(-180,180)).degrees 
    #        for r in triand["ra"]]
    #ax.scatter(ra, triand["dec"], d.to(u.kpc).value)
    ax.scatter(l, d.to(u.kpc).value)
    #ax.set_xlabel("RA [deg]")
    ax.set_xlabel("l [deg]")
    ax.set_ylabel("Dist. [kpc]")
    #ax.set_zlabel("Dist. [kpc]")
    
    return fig

def prob_rr_lyr_in_sgr():
    """ ambient halo density model is:   
            rho(R,Z) = [R^2 + (Z/q)^2]^(n/2)
    """
    
    lm10 = LM10().data
    lm10 = lm10[lm10["Pcol"] > -1]
    lm10 = lm10[lm10["dist"] < 100]
    
    values = np.vstack([lm10["ra"], lm10["dec"], lm10["dist"]])
    kernel = gaussian_kde(values)    
    
    # real data
    linear_sgr = ascii.read("data/catalog/LINEAR_RRab_sgr.txt")
    positions = np.vstack([linear_sgr["ra"],linear_sgr["dec"],linear_sgr["dist"]])
    
    probs = kernel(positions)
    fig,ax = plt.subplots(1, 1, figsize=(14,12))
    ax.plot(np.arange(len(probs)),probs,marker="o", linestyle="none")
    #plt.savefig("plots/test.png")
    
    members = linear_sgr[probs > 5E-7]
    fig,ax = plt.subplots(1, 1, figsize=(14,12))
    ax.plot(members["ra"], members["dist"], color="k", marker="o", linestyle="none")
    #plt.savefig("plots/test2.png")
    

if __name__ == "__main__":
    #prob_rr_lyr_in_sgr()
    sgr_kde()
