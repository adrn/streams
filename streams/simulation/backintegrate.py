# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

from streams.potential import *
from streams.simulation import Particle, TestParticleSimulation

def _variance_statistic(potential, xs, vs, sgr_cen):
    """ Compute the variance scalar that we will minimize.
        
        Parameters
        ----------
        potential : Potential
            The full Milky Way potential object.
        xs : ndarray
            An array of positions for all particles at all times
        vs : ndarray
            An array of velocities for all particles at all times
        sgr_cen : SgrCen
            Data for the Sgr satellite center, interpolated onto the
            time grid for our particles.
    """
    
    # Define tidal radius, escape velocity for satellite
    msat = 2.5E8 # M_sun
    sgr_orbital_radius = np.sqrt(sgr_cen.x**2 + sgr_cen.y**2 + sgr_cen.z**2)
    m_halo_enclosed = potential.params["v_halo"]**2 * sgr_orbital_radius/potential.params["_G"]
    mass_enclosed = potential.params["M_disk"] + potential.params["M_sph"] + m_halo_enclosed

    r_tides = sgr_orbital_radius * (msat / mass_enclosed)**(1./3)
    v_escs = np.sqrt(potential.params["_G"] * msat / r_tides)
    
    # N = number of particles
    N = xs.shape[1]
    
    # Distance to satellite center and total velocity
    d = np.sqrt((xs[:,:,0] - sgr_cen.x[:,np.newaxis].repeat(N, axis=1))**2 +
                (xs[:,:,1] - sgr_cen.y[:,np.newaxis].repeat(N, axis=1))**2 + 
                (xs[:,:,2] - sgr_cen.z[:,np.newaxis].repeat(N, axis=1))**2) / r_tides[:,np.newaxis].repeat(N, axis=1)
    
    v = np.sqrt((vs[:,:,0] - sgr_cen.vx[:,np.newaxis].repeat(N, axis=1))**2 +
                (vs[:,:,1] - sgr_cen.vy[:,np.newaxis].repeat(N, axis=1))**2 +
                (vs[:,:,2] - sgr_cen.vz[:,np.newaxis].repeat(N, axis=1))**2) / v_escs[:,np.newaxis].repeat(N, axis=1)
    
    idx = np.argmin(d**2 + v**2, axis=0)
    
    min_ds = []
    for ii,jj in enumerate(idx):
        min_ds.append(d[jj,ii])
    
    min_vs = []
    for ii,jj in enumerate(idx):
        min_vs.append(v[jj,ii])
    
    return np.var(min_ds) + np.var(min_vs)

def back_integrate(potential, sgr_snap, sgr_cen, dt):
    """ Given the particle snapshot information and a potential, integrate the particles
        backwards and return the variance scalar.
    """

    # Initialize particle simulation with full potential
    simulation = TestParticleSimulation(potential=potential)

    for ii in range(sgr_snap.num):
        p = Particle(position=(sgr_snap.data["x"][ii], sgr_snap.data["y"][ii], sgr_snap.data["z"][ii]), # kpc
                     velocity=(sgr_snap.data["vx"][ii], sgr_snap.data["vy"][ii], sgr_snap.data["vz"][ii]), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)

    ts, xs, vs = simulation.run(t1=max(sgr_cen.data["t"]), t2=min(sgr_cen.data["t"]), dt=-dt)

    return _variance_statistic(potential, xs, vs, sgr_cen)

def back_integrate_with_errors(potential, sgr_snap, sgr_cen, dt):
    """ Given the particle snapshot information and a potential, integrate the particles
        backwards WITH "realistic" observational uncertainties.
    """

    # Initialize particle simulation with full potential
    simulation = TestParticleSimulation(potential=potential)
       
    # Transform to heliocentric coordinates
    rsun = 8. # kpc
    
    # Distances in kpc, velocities in kpc/Myr
    x, y, z = sgr_snap.data["x"] + rsun, sgr_snap.data["y"], sgr_snap.data["z"]
    vx, vy, vz = sgr_snap.data["vx"], sgr_snap.data["vy"], sgr_snap.data["vz"]
    
    d = np.sqrt(x**2 + y**2 + z**2)
    vr = (x*vx + y*vy + z*vz) / d 
    
    # proper motions in km/s/kpc
    rad = np.sqrt(x**2 + y**2)
    vrad = (x*vx + y*vy) / rad
    mul = (x*vy - y*vx) / rad / d
    mub = (-z*vrad + rad*vz) / d**2
    
    # angular position
    sinb = z/d
    cosb = rad/d
    cosl = x/rad
    sinl = y/rad
    
    # DISTANCE ERROR -- assuming 2% distances from RR Lyrae mid-IR
    d += np.random.normal(0., 0.02*d)
    
    # VELOCITY ERROR -- 5 km/s (TODO: ???)
    vr += np.random.normal(0., (5.*u.km/u.s).to(u.kpc/u.Myr).value)
    
    # PROPER MOTION ERROR -- from GAIA
    # http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1
    #        25 muas at V=15 (RR Lyraes are ~A-F type :between B and G star)
    #        300 muas at V=20
    # with THANKS to Horace Smith
    # V abs mag RR Lyrae's
    #  M_v = (0.214 +/- 0.047)([Fe/H] + 1.5) + 0.45+/-0.05
    # Benedict et al. (2011 AJ, 142, 187)
    # assuming [Fe/H]=-0.5
    Mabs = 0.65
    
    # Johnson/Cousins (V-IC)  
    # (V-IC) color=vmic
    # 0.1-0.58
    # Guldenschuh et al. (2005 PASP 117, 721)
    vmic = 0.3
    V = Mabs + 5.*np.log10(d*100.)
    
    # GAIA G mag
    g = V - 0.0257 - 0.0924*vmic- 0.1623*vmic**2 + 0.0090*vmic**3
    zz = 10**(0.4*(g-15.)) # ???
    p = g < 12.
    
    if sum(p) > 0:
        zz[p] = 10**(0.4*(12. - 15.))
    
    # "end of mission parallax standard"
    # σπ [μas] = sqrt(9.3 + 658.1 · z + 4.568 · z^2) · [0.986 + (1 - 0.986) · (V-IC)]
    dp = np.sqrt(9.3 + 658.1*zz + 4.568*zz**2)*(0.986 + (1 - 0.986)*vmic)
    
    # assume 5 year baseline, mas/year
    dmu = dp/5.
    # too optimistic: following suggests factor 2 more realistic
    #http://www.astro.utu.fi/~cflynn/galdyn/lecture10.html 
    # - and Sanjib suggests factor 0.526
    dmu = 0.526*dp
    
    # translate to radians/year
    conv1 = np.pi/180./60./60./1.e6
    # translate to km/s from  kpc/year 
    kmpkpc =3.085678e16
    secperyr = 3.1536e7 
    conv2 = kmpkpc/secperyr
    dmu = dmu*conv1*conv2
    
    mul += np.random.normal(0., dmu)
    mub += np.random.normal(0., dmu)
    
    # CONVERT BACK
    x = d*cosb*cosl - rsun
    y = d*cosb*sinl
    z = d*sinb
    
    vx = vr*cosb*cosl - d*mul*sinl - d*mub*sinb*cosl
    vy = vr*cosb*sinl + d*mul*cosl - d*mub*sinb*sinl
    vz = vr*sinb + d*mub*cosb
    
    for ii in range(sgr_snap.num):
        p = Particle(position=(x[ii], y[ii], z[ii]), # kpc
                     velocity=(vx[ii], vy[ii] ,vz[ii]), # kpc/Myr
                     mass=1.) # M_sol
        simulation.add_particle(p)

    ts, xs, vs = simulation.run(t1=max(sgr_cen.data["t"]), t2=min(sgr_cen.data["t"]), dt=-dt)
    
    return _variance_statistic(potential, xs, vs, sgr_cen)