# coding: utf-8

""" MW Potential used in Law & Majewski 2010 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

from .common import MiyamotoNagaiPotential, HernquistPotential, LogarithmicPotentialLJ

halo_params = dict(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                    q1=1.38,
                    q2=1.0,
                    qz=1.36,
                    phi=1.692969*u.radian,
                    r_halo=12.*u.kpc)

param_ranges = dict(qz=(1.,2.),
                    q1=(1.,2.),
                    q2=(1.,2.),
                    v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr),
                            (150.*u.km/u.s).to(u.kpc/u.Myr)),
                    phi=(np.pi/4, 3*np.pi/4)*u.radian,
                    r_halo=(8,20)*u.kpc)

param_to_latex = dict(q1=r"$q_1$",
                      q2=r"$q_2$",
                      qz=r"$q_z$",
                      v_halo=r"$v_{halo}$",
                      phi=r"$\phi$",
                      r_halo=r"$r_{halo}$"
                      )

def LawMajewski2010(**halo_parameters):
    """ Construct the Milky Way gravitational potential used by 
        Law & Majewski 2010 for their Nbody simulation with Sgr.
    """
    
    if len(halo_parameters) == 0:
        halo_parameters = halo_params
    
    bases = [u.radian, u.Myr, u.kpc, u.M_sun]
    disk_potential = MiyamotoNagaiPotential(bases, m=1E11*u.M_sun,
                                            a=6.5*u.kpc,
                                            b=0.26*u.kpc)
    bulge_potential = HernquistPotential(bases, m=3.4E10*u.M_sun,
                                         c=0.7*u.kpc)
    
    params = halo_params.copy()
    for key,val in halo_parameters.items():
        params[key] = val
        
    halo_potential = LogarithmicPotentialLJ(bases, **params)
    
    return disk_potential + bulge_potential + halo_potential