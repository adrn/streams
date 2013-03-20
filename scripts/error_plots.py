# coding: utf-8

""" Make error plots for ∆d/d for GAIA and proper motion errors 
    vs. distance.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from streams.data.gaia import parallax_error, proper_motion_error, \
                              rr_lyrae_V_minus_I, rr_lyrae_M_V, \
                              apparent_magnitude

def plot_gaia_distance_error(ax):
    dp = parallax_error(m_V, rr_lyrae_V_minus_I).to(u.arcsecond).value
    dD = D.to(u.pc).value**2 * dp * u.pc # now error in parsecs?
    
    ax.loglog(D, (dD/D).decompose(), color="k", linewidth=1, alpha=0.5)
    ax.axhline(0.02, color="#3182BD", linewidth=2, linestyle="--")
    
    # Add vertical line at 30 kpc
    ax.axvline(30., color="#aaaaaa", linewidth=1)

def plot_vtan_error(ax):
    dpm = proper_motion_error(m_V, rr_lyrae_V_minus_I)
    dVtan = (dpm*D).to(u.km*u.radian/u.s).value
    
    ax.loglog(D, dVtan, color="k", linewidth=1, alpha=0.5)
    
    # Add vertical line at 30 kpc
    ax.axvline(30., color="#aaaaaa", linewidth=1)

def plot_proper_motion_error(ax):
    # HELL YEA! check this out: http://www.astro.utu.fi/~cflynn/galdyn/lecture10.html
    
    # proper motion error in mas/yr
    dpm = proper_motion_error(m_V, rr_lyrae_V_minus_I).to(u.arcsecond/u.year).value * 1000
    
    ax.loglog(D, dpm, color="k", linewidth=1, alpha=0.5)
    
    # Add vertical line at 30 kpc
    ax.axvline(30., color="#aaaaaa", linewidth=1)

if __name__ == "__main__":
    fig, axes = plt.subplots(3,1,figsize=(12,16))
    for fe_h in np.linspace(-1.6, 0.0, 25):
        # Assuming [Fe/H] for Sgr is -0.5
        M_V, dM_V = rr_lyrae_M_V(fe_h=fe_h, dfe_h=0.)
        
        # Distance from 100pc to ~50kpc
        D = np.logspace(-1., 1.6, 50)*u.kpc
        
        # Compute the apparent magnitude as a function of distance
        m_V = apparent_magnitude(M_V, D)
    
        plot_gaia_distance_error(axes[0])
        plot_vtan_error(axes[1])
        plot_proper_motion_error(axes[2])
    
    axes[0].set_title("GAIA parallax distance uncertainty for Sgr RR Lyrae")
    axes[0].set_ylabel(r"Fractional distance error, $\sigma_D/D$")
    axes[0].annotate('2% Spitzer distance', xy=(np.mean(D), 0.019),  xycoords='data',
                    xytext=(-50, -50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3,rad=.2")
                    )
    
    axes[1].set_title(r"GAIA $v_{tan}$ uncertainty for Sgr RR Lyrae")
    axes[1].set_ylabel(r"Fractional $v_{tan}$ error, $\sigma_\mu\times D$ [km/s]")
    
    axes[2].set_title("GAIA proper motion uncertainty for Sgr RR Lyrae")
    axes[2].set_xlabel("Distance [kpc]")
    axes[2].set_ylabel(r"$\sigma_\mu$ [mas/yr]")
    
    axes[0].set_xlim(0.1, 50.)
    axes[1].set_xlim(0.1, 50.)
    axes[2].set_xlim(0.1, 50.)
    
    fig.savefig("plots/figures/gaia_errors.png")