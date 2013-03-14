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

from streams.data.gaia import parallax_error, proper_motion_error

D = np.logspace(0., 4.6, 50)*u.pc
dp = parallax_error(D).to(u.arcsecond).value
dD = D.to(u.pc).value**2 * dp

plt.figure(figsize=(8,8))
ax = plt.loglog(D, dD/D, color="k", linewidth=2)
plt.axhline(0.02, color="#3182BD", linewidth=2, linestyle="--")

# Add vertical line at 30 kpc
plt.axvline(30000., color="#aaaaaa", linewidth=1)

plt.title("GAIA parallax distance uncertainty for Sgr RR Lyrae")
plt.xlabel("Distance [pc]")
plt.ylabel(r"Fractional distance error, $\sigma_D/D$")
plt.annotate('2% Spitzer distance', xy=(np.mean(D), 0.019),  xycoords='data',
                xytext=(-50, -50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2")
                )
plt.tight_layout()
plt.savefig("plots/figures/gaia_distance_error.png")

# ------------------------------------------------------------------------

dpm = proper_motion_error(D)
dVtan = (dpm*D).to(u.km*u.radian/u.s).value

plt.figure(figsize=(8,8))
plt.loglog(D, dVtan, color="k", linewidth=2)

# Add vertical line at 30 kpc
D_30 = np.array([30])*u.kpc
dpm = proper_motion_error(D_30)
dVtan = (dpm*D_30).to(u.km*u.radian/u.s).value
plt.axvline(30000., color="#aaaaaa", linewidth=1)
plt.axhline(dVtan[0], color="#aaaaaa", linewidth=1)

plt.title(r"GAIA $v_{tan}$ uncertainty for Sgr RR Lyrae")
plt.xlabel("Distance [pc]")
plt.ylabel(r"Fractional $v_{tan}$ error, $\sigma_\mu\times D$ [km/s]")
plt.tight_layout()
plt.savefig("plots/figures/gaia_vtan_error.png")

# ------------------------------------------------------------------------

dpm = proper_motion_error(D).to(u.arcsecond/u.year).value * 1000

plt.figure(figsize=(8,8))
plt.loglog(D, dpm, color="k", linewidth=2)

# Add vertical line at 30 kpc
plt.axvline(30000., color="#aaaaaa", linewidth=1)

plt.title("GAIA proper motion uncertainty for Sgr RR Lyrae")
plt.xlabel("Distance [pc]")
plt.ylabel(r"$\sigma_\mu$ [mas/yr]")
plt.tight_layout()
plt.savefig("plots/figures/gaia_proper_motion_error.png")