# coding: utf-8

""" In this module, I'll test how the distribution of 'energy distance' changes as I tweak
    various galaxy potential parameters. Ultimately, I want to come up with a way to evaluate
    the 'best' potential.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
from scipy import interpolate
import matplotlib.pyplot as plt
import astropy.units as u
import emcee

# Project
from streams.data import SgrSnapshot, SgrCen
from streams.potential import *
from streams.integrate import leapfrog
from streams.simulation import Particle, ParticleSimulation

def plot_projections(x, y, z, axes=None, **kwargs):
    """ Make a scatter plot of particles in projections of the supplied coordinates.
        Extra kwargs are passed to matplotlib's scatter() function.
    """

    if axes == None:
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    else:
        fig = axes[0,0].figure

    axes[0,1].set_visible(False)

    axes[0,0].scatter(x, y, **kwargs)
    axes[0,0].set_ylabel("Y [kpc]")

    axes[1,0].scatter(x, z, **kwargs)
    axes[1,0].set_xlabel("X [kpc]")
    axes[1,0].set_ylabel("Z [kpc]")

    axes[1,1].scatter(y, z, **kwargs)
    axes[1,0].set_xlabel("Y [kpc]")

    return fig, axes

# --------------------------------------------------------------------
# This could get messy (TM)
# --------------------------------------------------------------------

# Read in data from Kathryn's SGR_SNAP and SGR_CEN files
sgr_cen = SgrCen()
dt = sgr_cen.data["dt"][0]*10.
np.random.seed(42)
sgr_snap = SgrSnapshot(num=100, no_bound=True) # randomly sample 100 particles

# Get timestep information from SGR_CEN
t1 = min(sgr_cen.data["t"])
t2 = max(sgr_cen.data["t"])
ts = np.arange(t2, t1, -dt)

true_halo_params = dict(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value,
                        q1=1.38,
                        q2=1.0,
                        qz=1.36,
                        phi=1.692969,
                        c=12.)

def ln_p_qz(qz):
    """ Prior on vertical (z) axis ratio """
    if qz <= 1 or qz >= 2:
        return -np.inf
    else:
        return 0.

def ln_p_q1(q1):
    """ Prior on axis ratio """
    if q1 <= 1 or q1 >= 2:
        return -np.inf
    else:
        return 0.

def ln_p_q2(q2):
    """ Prior on axis ratio """
    if q2 <= 0.5 or q2 >= 2:
        return -np.inf
    else:
        return 0.

def ln_p_v_halo(v):
    """ Prior on mass of the halo (v_halo). The range imposed is roughly a
        halo mass between 10^10 and 10^12 M_sun at 200 kpc
    """
    if v <= 0.01 or v >= 0.15:
        return -np.inf
    else:
        return 0.

def ln_p_phi(phi):
    """ Prior on orientation angle between DM halo and disk """
    if phi < 0. or phi > np.pi:
        return -np.inf
    else:
        return 0.

def ln_p_c(c):
    """ Prior on halo concentration parameter """
    if c < 5. or c > 20:
        return -np.inf
    else:
        return 0.

def ln_prior(p):
    return ln_p_qz(p[0]) + ln_p_v_halo(p[1]) + ln_p_c(p[2]) + ln_p_phi(p[3]) #+ ln_p_q2(p[2]) + ln_p_q1(p[1])

def ln_likelihood(p):
    # sgr_snap are the data!

    halo_params = true_halo_params.copy()
    halo_params["qz"] = p[0]
    #halo_params["q1"] = p[1]
    halo_params["v_halo"] = p[1]
    halo_params["c"] = p[2]
    halo_params["phi"] = p[3]
    halo_potential = LogarithmicPotentialLJ(**halo_params)

    energy_distances = run_back_integration(halo_potential, sgr_snap)
    return -np.mean(energy_distances)

def ln_posterior(p):
    return ln_prior(p) + ln_likelihood(p)

def infer_potential():
    nwalkers = 12
    param_names = ["qz", "v_halo", "c", "phi"]
    p0 = np.array([[np.random.uniform(1,2),
                    np.random.uniform(0.01,0.15),
                    np.random.uniform(5,20),
                    np.random.uniform(0., np.pi)] for ii in range(nwalkers)])
    ndim = p0.shape[1]

    nthreads = 4
    nburn_in = 100
    nsamples = 1000

    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior,
                                    threads=nthreads)
    pos, prob, state = sampler.run_mcmc(p0, nburn_in)
    print("Burn in complete...")
    sampler.reset()
    sampler.run_mcmc(pos, nsamples)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    fig,axes = plt.subplots(ndim, 1, figsize=(14,5*(ndim+1)))
    fig.suptitle("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    for ii in range(ndim):
        axes[ii].set_title(param_names[ii])
        axes[ii].hist(sampler.flatchain[:,ii], bins=25, color="k", histtype="step", alpha=0.75)
        axes[ii].axvline(true_halo_params[param_names[ii]], color="k", linestyle="--", linewidth=2)

    plt.savefig("plots/posterior.png")

    return

# --------------------------------------------------
# Define tidal radius, escape velocity for satellite
# --------------------------------------------------

# First I have to interpolate the SGR_CEN data so we can evaluate the position at each particle timestep
cen_x = interpolate.interp1d(sgr_cen.data["t"], sgr_cen.data["x"], kind='cubic')(ts)
cen_y = interpolate.interp1d(sgr_cen.data["t"], sgr_cen.data["y"], kind='cubic')(ts)
cen_z = interpolate.interp1d(sgr_cen.data["t"], sgr_cen.data["z"], kind='cubic')(ts)
cen_vx = interpolate.interp1d(sgr_cen.data["t"], sgr_cen.data["vx"], kind='cubic')(ts)
cen_vy = interpolate.interp1d(sgr_cen.data["t"], sgr_cen.data["vy"], kind='cubic')(ts)
cen_vz = interpolate.interp1d(sgr_cen.data["t"], sgr_cen.data["vz"], kind='cubic')(ts)

def run_back_integration(halo_potential, sgr_snap):
    """ Given the particle snapshot information and a potential, integrate the particles
        backwards and return the minimum energy distances.
    """

    # We use the same disk and bulge potentials for all runs, just vary the halo potential
    disk_potential = MiyamotoNagaiPotential(M=1E11*u.M_sun,
                                            a=6.5,
                                            b=0.26)
    bulge_potential = HernquistPotential(M=3.4E10*u.M_sun,
                                         c=0.7)
    potential = disk_potential + bulge_potential + halo_potential

    # Initialize particle simulation with full potential
    simulation = ParticleSimulation(potential=potential)

    for ii in range(sgr_snap.num):
        p = Particle(position=(sgr_snap.data["x"][ii], sgr_snap.data["y"][ii], sgr_snap.data["z"][ii]), # kpc
                     velocity=(sgr_snap.data["vx"][ii], sgr_snap.data["vy"][ii], sgr_snap.data["vz"][ii]), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)

    # The data in SGR_CEN is only printed every 25 steps!
    ts, xs, vs = simulation.run(t1=t2, t2=t1, dt=-dt)

    msat = 2.5E8 # M_sun
    sgr_orbital_radius = np.sqrt(cen_x**2 + cen_y**2 + cen_z**2)
    m_halo_enclosed = halo_potential.params["v_halo"]**2 * sgr_orbital_radius/bulge_potential.params["_G"]
    mass_enclosed = disk_potential.params["M"] + bulge_potential.params["M"] + m_halo_enclosed

    r_tides = sgr_orbital_radius * (msat / mass_enclosed)**(1./3)
    v_escs = np.sqrt(bulge_potential.params["_G"] * msat / r_tides)

    closest_distances = []
    for ii in range(sgr_snap.num):
        # Distance to satellite center and total velocity
        d = np.sqrt((xs[:,ii,0] - cen_x)**2 +
                    (xs[:,ii,1] - cen_y)**2 +
                    (xs[:,ii,2] - cen_z)**2)
        v = np.sqrt((vs[:,ii,0] - cen_vx)**2 +
                    (vs[:,ii,1] - cen_vy)**2 +
                    (vs[:,ii,2] - cen_vz)**2)

        energy_distances = np.sqrt((d/r_tides)**2 + (v/v_escs)**2)
        closest_distances.append(min(energy_distances))

    return np.array(closest_distances)

if __name__ == "__main__":
    infer_potential()
