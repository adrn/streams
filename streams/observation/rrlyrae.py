# coding: utf-8

""" Handling RR Lyrae data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable

__all__ = ["rrl_M_V", "rrl_photometric_distance", "rrl_V_minus_I"]

# Johnson/Cousins (V - I_C) color for RR Lyrae at *minimum*
# Guldenschuh et al. (2005 PASP 117, 721), pg. 725
# (V-I)_min = 0.579 +/- 0.006 mag
rrl_V_minus_I = 0.579

def rrl_M_V(fe_h, dfe_h=0.):
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

def rrl_photometric_distance(m_V, fe_h):
    """ Estimate the distance to an RR Lyrae given its apparent V-band
        magnitude and metallicity.
    """
    M_V, dM_V = rrl_M_V(fe_h)
    mu = m_V - M_V
    
    d = 10**(mu/5. + 1) * u.pc
    
    return d.to(u.kpc)