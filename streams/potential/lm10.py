# coding: utf-8

""" MW Potential used in Law & Majewski 2010 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

from .core import CartesianPotential, CompositePotential, UnitSystem
from .common import MiyamotoNagaiPotential, HernquistPotential, LogarithmicPotentialLJ
from ._lm10_acceleration import lm10_acceleration

true_params = dict(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                   q1=1.38,
                   q2=1.0,
                   qz=1.36,
                   phi=(97.*u.degree).to(u.radian),
                   r_halo=12.*u.kpc)

param_units = dict(v_halo=u.kpc/u.Myr,
                   q1=1.,
                   q2=1.,
                   qz=1.,
                   phi=u.radian,
                   r_halo=u.kpc)

param_to_latex = dict(q1=r"$q_1$",
                      q2=r"$q_2$",
                      qz=r"$q_z$",
                      v_halo=r"$v_{\rm halo}$",
                      phi=r"$\phi$",
                      r_halo=r"$r_{\rm halo}$"
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
        self._G = G.decompose(bases=unit_system).value
        
    def _tidal_radius(self, m, r):
        """ Compute the tidal radius of a massive particle at the specified 
            position(s). Assumes position and mass are in the same unit 
            system as the potential.
            
            Parameters
            ----------
            m : numeric
                Mass.
            r : array_like
                Position.
        """
        
        # Radius of Sgr center relative to galactic center
        R_orbit = np.sqrt(np.sum(r**2., axis=-1)) 
        
        m_halo_enc = self["halo"]._parameters["v_halo"]**2 * R_orbit/self._G
        m_enc = self["disk"]._parameters["m"] + \
                self["bulge"]._parameters["m"] + \
                m_halo_enc
        
        return R_orbit * (m / (3.*m_enc))**(0.33333)
    
    def tidal_radius(self, m, r):
        """ Compute the tidal radius of a massive particle at the specified 
            position(s). 
            
            Parameters
            ----------
            m : astropy.units.Quantity
                Mass.
            r : astropy.units.Quantity
                Position.
        """
        
        if not hasattr(r, "decompose") or not hasattr(m, "decompose"):
            raise TypeError("Position and mass must be Quantity objects.")
        
        R_tide = self._tidal_radius(r=r.decompose(self.unit_system).value,
                                    m=m.decompose(self.unit_system).value)
        
        return R_tide * self.unit_system['length']

    def _escape_velocity(self, m, r=None, r_tide=None):
        """ Compute the escape velocity of a satellite in a potential given
            its tidal radius. Assumes position and mass are in the same unit 
            system as the potential.
            
            Parameters
            ----------
            m : numeric
                Mass.
            r : array_like
                Position.
            or
            r_tide : array_like
                Tidal radius.
        """
        
        if r is not None and r_tide is None:
            r_tide = self._tidal_radius(m, r)
        
        elif r_tide is not None and r is None:
            pass
        
        else:
            raise ValueError("Must specify just r or r_tide.")
        
        return np.sqrt(2. * self._G * m / r_tide)
    
    def escape_velocity(self, m, r=None, r_tide=None):
        """ Compute the escape velocity of a satellite in a potential given
            its tidal radius. 
            
            Parameters
            ----------
            m : astropy.units.Quantity
                Mass.
            r : astropy.units.Quantity
                Position.
            or
            r_tide : astropy.units.Quantity
                Tidal radius.
        """
        
        if not hasattr(m, "decompose"):
            raise TypeError("Mass must be a Quantity object.")
        
        if r is not None and r_tide is None:
            r_tide = self.tidal_radius(m, r)
        
        elif r_tide is not None and r is None:
            if not hasattr(r_tide, "decompose"):
                raise TypeError("r_tide must be a Quantity object.")
        
        else:
            raise ValueError("Must specify just r or r_tide.")
        
        v_esc = self._escape_velocity(m=m.decompose(self.unit_system).value,
                                      r_tide=r_tide.decompose(self.unit_system).value)
        
        return v_esc * self.unit_system['length'] / self.unit_system['time']