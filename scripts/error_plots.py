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

from streams.data.gaia import parallax_error

D = np.logspace(0., 4.6, 50)*u.pc
dp = parallax_error(D).to(u.arcsecond).value
dD = D.to(u.pc).value**2 * dp

plt.figure(figsize=(8,8))
plt.loglog(D, dD/D, color="k", linewidth=2)
plt.axhline(0.02, color="#3182BD", linewidth=2, linestyle="--")
plt.xlabel("Distance [pc]")
plt.ylabel(r"Fractional distance error [$\sigma_D/D$]")
plt.tight_layout()
plt.savefig("plots/figures/gaia_distance_error.png")