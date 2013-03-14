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

def parallax_error(d):
    """ http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance#chapter1
            25 muas at V=15 (RR Lyraes are ~A-F type :between B and G star)
            300 muas at V=20
        with thanks to Horace Smith
    """
    if not isinstance(d, u.Quantity):
        raise TypeError("Distance array must be an Astropy Quantity object!")
    
    # V abs mag RR Lyrae's
    #  M_v = (0.214 +/- 0.047)([Fe/H] + 1.5) + 0.45+/-0.05
    # Benedict et al. (2011 AJ, 142, 187)
    # assuming [Fe/H] = -0.5 for Sgr
    fe_h = -0.5
    Mabs = 0.214*(fe_h + 1.5) + 0.45
    
    # Johnson/Cousins (V-IC)  
    # (V-IC) color=vmic
    # 0.1-0.58
    # Guldenschuh et al. (2005 PASP 117, 721)
    vmic = 0.3
    V = Mabs + 5.*np.log10(d.to(u.pc).value)
    
    # GAIA G mag
    g = V - 0.0257 - 0.0924*vmic- 0.1623*vmic**2 + 0.0090*vmic**3
    zz = 10**(0.4*(g-15.)) 
    p = g < 12.
    
    if sum(p) > 0:
        zz[p] = 10**(0.4*(12. - 15.))
    
    # "end of mission parallax standard"
    # σπ [μas] = sqrt(9.3 + 658.1 · z + 4.568 · z^2) · [0.986 + (1 - 0.986) · (V-IC)]
    dp = np.sqrt(9.3 + 658.1*zz + 4.568*zz**2)*(0.986 + (1 - 0.986)*vmic)*1E-6*u.arcsecond
    
    return dp

def proper_motion_error(d):
    dp = parallax_error(d)
    
    # assume 5 year baseline, µas/year
    dmu = dp/(5.*u.year)
    
    # too optimistic: following suggests factor 2 more realistic
    #http://www.astro.utu.fi/~cflynn/galdyn/lecture10.html 
    # - and Sanjib suggests factor 0.526
    dmu = 0.526*dmu

    return dmu.to(u.radian/u.yr)