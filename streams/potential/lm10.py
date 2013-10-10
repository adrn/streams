# coding: utf-8

""" MW Potential used in Law & Majewski 2010 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import math

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

from .core import CartesianPotential, CompositePotential
from .common import MiyamotoNagaiPotential, HernquistPotential, LogarithmicPotentialLJ
from ._lm10_acceleration import lm10_acceleration

# v_halo range comes from 5E11 < M < 5E12, current range of MW mass @ 200 kpc
param_ranges = dict(v_halo=((100.*u.km/u.s).to(u.kpc/u.Myr).value,
                            (330.*u.km/u.s).to(u.kpc/u.Myr).value),
                    q1=(1.,2.),
                    q2=(0.5,2.),
                    qz=(1.0,2.),
                    phi=(np.pi/4, 3*np.pi/4),
                    r_halo=(8,20)) # kpc

true_params = dict(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                   q1=1.38,
                   q2=1.0,
                   qz=1.36,
                   phi=(97.*u.degree).to(u.radian),
                   r_halo=12.*u.kpc)

_true_params = dict(v_halo=0.124625659009,
                   q1=1.38,
                   q2=1.0,
                   qz=1.36,
                   phi=1.69296937443,
                   r_halo=12.)

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
        
        units = (u.kpc, u.Myr, u.radian, u.M_sun)
        
        for p in ["q1", "q2", "qz", "phi", "v_halo", "r_halo"]:
            if p not in parameters.keys():
                parameters[p] = true_params[p]
        
        bulge = HernquistPotential(units,
                                   m=3.4E10*u.M_sun,
                                   c=0.7*u.kpc)
                                   
        disk = MiyamotoNagaiPotential(units,
                                      m=1.E11*u.M_sun, 
                                      a=6.5*u.kpc,
                                      b=0.26*u.kpc)
        halo = LogarithmicPotentialLJ(units,
                                      **parameters)
        
        super(LawMajewski2010, self).__init__(units, 
                                              bulge=bulge,
                                              disk=disk,
                                              halo=halo)
        
        _params = halo._parameters.copy()
        _params.pop('r_0')
        
        self._acceleration_at = lambda r, n_particles, acc: lm10_acceleration(r, n_particles, acc, **_params)
        self._G = G.decompose(bases=units).value
    
    def _enclosed_mass(self, R):
        """ Compute the enclosed mass at the position r. Assumes it's far from
            the disk and bulge.
        """ 
        
        m_halo_enc = self["halo"]._parameters["v_halo"]**2 * R/self._G
        m_enc = self["disk"]._parameters["m"] + \
                self["bulge"]._parameters["m"] + \
                m_halo_enc
        
        return m_enc
    
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
        m_enc = self._enclosed_mass(R_orbit)
        
        return R_orbit * (m / (m_enc))**(0.33333)
    
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
        
        R_tide = self._tidal_radius(r=r.decompose(self.units).value,
                                    m=m.decompose(self.units).value)
        
        return R_tide * r.unit

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
        
        v_esc = self._escape_velocity(m=m.decompose(self.units).value,
                                      r_tide=r_tide.decompose(self.units).value)
        
        r_unit = filter(lambda x: x.is_equivalent(u.km), units)[0]
        t_unit = filter(lambda x: x.is_equivalent(u.s), units)[0]
        return v_esc * r_unit/t_unit