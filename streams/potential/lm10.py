# coding: utf-8

""" MW Potential used in Law & Majewski 2010 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import uuid

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

from .core import Potential
from .common import MiyamotoNagaiPotential, HernquistPotential, LogarithmicPotentialLJ

halo_params = dict(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value,
                    q1=1.38,
                    q2=1.0,
                    qz=1.36,
                    phi=1.692969,
                    r_halo=12.)

param_ranges = dict(qz=(1.,2.),
                    q1=(1.,2.),
                    q2=(1.,2.),
                    v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr).value,
                            (150.*u.km/u.s).to(u.kpc/u.Myr).value),
                    phi=(np.pi/4, 3*np.pi/4),
                    r_halo=(8,20))

def LawMajewski2010(**halo_parameters):
    """ Construct the Milky Way gravitational potential used by 
        Law & Majewski 2010 for their Nbody simulation with Sgr.
    """
    
    disk_potential = MiyamotoNagaiPotential(M=1E11*u.M_sun,
                                            a=6.5,
                                            b=0.26)
    bulge_potential = HernquistPotential(M=3.4E10*u.M_sun,
                                         c=0.7)
    
    params = halo_params.copy()
    for key,val in halo_parameters.items():
        params[key] = val
        
    halo_potential = LogarithmicPotentialLJ(**params)
    
    return disk_potential + bulge_potential + halo_potential