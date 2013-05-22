# coding: utf-8

""" MW Potential used in Law & Majewski 2010 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

from .core import CartesianPotential, CompositePotential, UnitSystem
from .common import MiyamotoNagaiPotential, HernquistPotential, LogarithmicPotentialLJ
from ._lm10_acceleration import lm10_acceleration

true_params = dict(v_halo=(121.858*u.km/u.s),
                   q1=1.38,
                   q2=1.0,
                   qz=1.36,
                   phi=1.692969*u.radian,
                   r_halo=12.*u.kpc)

param_ranges = dict(qz=(1.,2.),
                    q1=(1.,2.),
                    q2=(1.,2.),
                    v_halo=((100.*u.km/u.s).value,(150.*u.km/u.s).value),
                    phi=(np.pi/4, 3*np.pi/4),
                    r_halo=(8,20)) # kpc

param_units = dict(v_halo=u.km/u.s,
                   q1=1.,
                   q2=1.,
                   qz=1.,
                   phi=u.radian,
                   r_halo=u.kpc)

param_to_latex = dict(q1=r"$q_1$",
                      q2=r"$q_2$",
                      qz=r"$q_z$",
                      v_halo=r"$v_{halo}$",
                      phi=r"$\phi$",
                      r_halo=r"$r_{halo}$"
                      )

class LawMajewski2010(CompositePotential):
    
    def __init__(self, **parameters):
        """ Represents the functional form of the Galaxy potential used by 
            Law and Majewski 2010.
            
            Miyamoto-Nagai disk
            Hernquist bulge
            Logarithmic halo
            
            Model parameters: q1, qz, phi, v_halo
            
            Parameters
            ----------
            parameters : dict
                A dictionary of parameters for the potential definition.
        """
        
        latex = ""
        
        unit_system = UnitSystem(u.kpc, u.Myr, u.radian, u.M_sun)
        unit_system = self._validate_unit_system(unit_system)
        
        for p in ["q1", "q2", "qz", "phi", "v_halo", "r_halo"]:
            if p not in parameters.keys():
                parameters[p] = true_params[p]
        
        bulge = HernquistPotential(unit_system,
                                   m=3.4E10*u.M_sun,
                                   c=0.7*u.kpc)
                                   
        disk = MiyamotoNagaiPotential(unit_system,
                                      m=1.E11*u.M_sun, 
                                      a=6.5*u.kpc,
                                      b=0.26*u.kpc)
        halo = LogarithmicPotentialLJ(unit_system,
                                      **parameters)
        
        super(LawMajewski2010, self).__init__(unit_system, 
                                              bulge=bulge,
                                              disk=disk,
                                              halo=halo)
        
        self._acceleration_at = lambda r: lm10_acceleration(r, **halo._parameters)
        
# NOTE: Now that I've implemented the _acceleration_at in the above, this Cython
#   potential is obsolete.
class CLawMajewski2010(CartesianPotential):
    
    def __init__(self, **parameters):
        """ Represents the functional form of the Galaxy potential used by 
            Law and Majewski 2010. 
            
            Note::
                For this function, the evaluation and acceleration functions
                are implemented in Cython!
            
            Miyamoto-Nagai disk
            Hernquist bulge
            Logarithmic halo
            
            Model parameters: q1, qz, phi, v_halo
            
            Parameters
            ----------
            parameters : dict
                A dictionary of parameters for the potential definition.
        """
        
        latex = ""
        
        unit_system = UnitSystem(u.kpc, u.Myr, u.radian, u.M_sun)
        unit_system = self._validate_unit_system(unit_system)
        
        for p in ["q1", "q2", "qz", "phi", "v_halo", "r_halo"]:
            if p not in parameters.keys():
                parameters[p] = true_params[p]
        
        parameters["r_0"] = [0.,0.,0.]*u.kpc
        
        # get functions for evaluating potential and derivatives
        f = lambda *args,**kwargs: None
        super(CLawMajewski2010, self).__init__(unit_system, 
                                               f=f, f_prime=lm10_acceleration, 
                                               latex=latex, 
                                               parameters=parameters)

    
    