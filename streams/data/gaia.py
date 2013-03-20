# coding: utf-8

""" Description... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable

__all__ = ["parallax_error", "proper_motion_error", "rr_lyrae_M_V", \
           "apparent_magnitude", "add_observational_uncertainties"]

def rr_lyrae_M_V(fe_h, dfe_h=0.):
    """ Given an RR Lyra metallicity, return the V-band absolute magnitude. 
        
        This expression comes from Benedict et al. 2011 (AJ 142, 187), 
        equation 14 reads:
            M_v = (0.214 +/- 0.047)([Fe/H] + 1.5) + a_7
        
        where
            a_7 = 0.45 +/- 0.05
            
        From that, we take the absolute V-band magnitude to be:
            Mabs = 0.214 * ([Fe/H] + 1.5) + 0.45
            δMabs = sqrt[(0.047*(δ[Fe/H]))**2 + (0.05)**2]
        
        Parameters
        ----------
        fe_h : numeric or iterable
            Metallicity.
        dfe_h : numeric or iterable
            Uncertainty in the metallicity.
        
    """
    
    if isiterable(fe_h):
        fe_h = np.array(fe_h)
        dfe_h = np.array(dfe_h)
        
        if not fe_h.shape == dfe_h.shape:
            raise ValueError("Shape mismatch: fe_h and dfe_h must have the same shape.")
    
    # V abs mag for RR Lyrae
    Mabs = 0.214*(fe_h + 1.5) + 0.45
    dMabs = np.sqrt((0.047*dfe_h)**2 + (0.05)**2)
    
    return (Mabs, dMabs)

def apparent_magnitude(M_V, d):
    """ Compute the apparent magnitude of a source given an absolute magnitude
        and a distance.
        
        Parameters
        ----------
        M_V : numeric or iterable
            Absolute V-band magnitude of a source.
        d : astropy.units.Quantity
            The distance to the source as a Quantity object.
            
    """
    
    if not isinstance(d, u.Quantity):
        raise TypeError("Distance must be an Astropy Quantity object!")
    
    # Compute the apparent magnitude -- ignores extinction
    return M_V - 5.*(1. - np.log10(d.to(u.pc).value))

def parallax_error(V, V_minus_I):
    """ Compute the estimated GAIA parallax error as a function of apparent 
        V-band magnitude and V-I color. All equations are taken from the GAIA
        science performance book:
    
            http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1 
    
        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """
    
    # GAIA G mag
    g = V - 0.0257 - 0.0924*V_minus_I- 0.1623*V_minus_I**2 + 0.0090*V_minus_I**3
    z = 10**(0.4*(g-15.)) 
    
    p = g < 12.
    if isiterable(V):
        if sum(p) > 0:
            z[p] = 10**(0.4*(12. - 15.))
    else:
        if p:
            z[p] = 10**(0.4*(12. - 15.))
    
    # "end of mission parallax standard"
    # σπ [μas] = (9.3 + 658.1·z + 4.568·z^2)^(1/2) · [0.986 + (1 - 0.986) · (V-IC)]
    dp = np.sqrt(9.3 + 658.1*z + 4.568*z**2) * (0.986 + (1 - 0.986)*V_minus_I) * 1E-6 * u.arcsecond
    
    return dp

def proper_motion_error(V, V_minus_I):
    """ Compute the estimated GAIA proper motion error as a function of apparent 
        V-band magnitude and V-I color. All equations are taken from the GAIA
        science performance book:
    
            http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1 
    
        Parameters
        ----------
        V : numeric or iterable
            The V-band apparent magnitude of a source.
        V_minus_I : numeric or iterable
            The V-I color of the source.
    """
    
    dp = parallax_error(V, V_minus_I)
    
    # assume 5 year baseline, µas/year
    dmu = dp/(5.*u.year)
    
    # too optimistic: following suggests factor 2 more realistic
    #http://www.astro.utu.fi/~cflynn/galdyn/lecture10.html 
    # - and Sanjib suggests factor 0.526
    dmu = 0.526*dmu

    return dmu.to(u.radian/u.yr)

def add_observational_uncertainties(x,y,z,vx,vy,vz):
    """ Given 3D galactocentric position and velocity, transform to heliocentric
        coordinates, apply observational uncertainty estimates, then transform
        back to galactocentric frame.
    """
    
    if not isinstance(x,u.Quantity) or not isinstance(y,u.Quantity) or not isinstance(z,u.Quantity):
        raise TypeError("Positions must be Astropy Quantity objects!")
    
    if not isinstance(vx,u.Quantity) or not isinstance(vy,u.Quantity) or not isinstance(vz,u.Quantity):
        raise TypeError("Velocities must be Astropy Quantity objects!")
    
    # Johnson/Cousins (V-I_C)
    # (V-I_C) color
    # 0.1-0.58
    # Guldenschuh et al. (2005 PASP 117, 721)
    rr_lyrae_V_minus_I = 0.3
    
    # assuming [Fe/H] = -0.5 for Sgr
    M_V = rr_lyrae_M_V(-0.5)
    
    # Transform to heliocentric coordinates
    rsun = 8.*u.kpc
    
    x += rsun
    
    d = np.sqrt(x**2 + y**2 + z**2)*x.unit
    V = apparent_magnitude(M_V, d)
    
    vr = (x*vx + y*vy + z*vz) / d 
    
    # proper motions in km/s/kpc
    rad = np.sqrt(x**2 + y**2)*x.unit
    vrad = (x*vx + y*vy) / rad
    mul = (x*vy - y*vx) / rad / d
    mub = (-z*vrad + rad*vz) / d**2
       
    # angular position
    sinb = z/d
    cosb = rad/d
    cosl = x/rad
    sinl = y/rad
    
    # DISTANCE ERROR -- assuming 2% distances from RR Lyrae mid-IR
    d += np.random.normal(0., 0.02*d.value)*d.unit
    
    # VELOCITY ERROR -- 5 km/s (TODO: ???)
    vr += np.random.normal(0., (5.*u.km/u.s))*u.km/u.s

    dmu = proper_motion_error(V, rr_lyrae_V_minus_I)
    
    # translate to radians/year
    #conv1 = np.pi/180./60./60./1.e6
    # translate to km/s from  kpc/year 
    #kmpkpc = 3.085678e16
    #secperyr = 3.1536e7 
    #conv2 = kmpkpc/secperyr
    #dmu = dmu*conv1*conv2
    
    dmu = (dmu.to(u.rad/u.s).value / u.s).to(u.km / (u.kpc*u.s))
    mul += np.random.normal(0., dmu.value)*dmu.unit
    mub += np.random.normal(0., dmu.value)*dmu.unit
    
    # CONVERT BACK
    x = d*cosb*cosl - rsun
    y = d*cosb*sinl
    z = d*sinb
    
    vx = vr*cosb*cosl - d*mul*sinl - d*mub*sinb*cosl
    vy = vr*cosb*sinl + d*mul*cosl - d*mub*sinb*sinl
    vz = vr*sinb + d*mub*cosb
    
    return (x,y,z,vx,vy,vz)