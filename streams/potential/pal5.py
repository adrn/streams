# coding: utf-8

""" MW Potential used in Andreas' Palomar 5 simulation """

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
from .common import MiyamotoNagaiPotential, HernquistPotential, \
                    AxisymmetricNFWPotential, AxisymmetricLogarithmicPotential
from ._pal5_acceleration import pal5_acceleration, pal5_logarithmic

'''
true_params = dict(log_m=28.2254523235*u.M_sun,
                   qz=0.814,
                   Rs=32.26*u.kpc)

_true_params = dict(log_m=28.2254523235,
                   qz=0.814,
                   Rs=32.26)
'''

true_params = dict(qz=0.814,
                   v_c=0.2454509*u.kpc/u.Myr)

_true_params = dict(v_c=0.2454509,
                    qz=0.814)

param_to_latex = dict(v_c=r"$v_c$",
                      qz=r"$q_z$")

param_ranges = dict(log_m=(26.93787, 29.24046),
                    qz=(0.707,1.2),
                    Rs=(10.,45.),
                    v_c=(0.1840881,0.286359406))

class Palomar5(CompositePotential):
    
    def __init__(self, **parameters):
        """ Represents the functional form of the Galaxy potential used by 
            Andreas' Palomar 5 simulation.
            
            Miyamoto-Nagai disk
            Hernquist bulge
            Axisymmetric, flattened NFW halo
            
            Model parameters: log_M, qz, Rs
            
            Parameters
            ----------
            parameters : dict
                A dictionary of parameters for the potential definition.
        """
        
        latex = ""
        
        units = (u.kpc, u.Myr, u.radian, u.M_sun)
        
        for p in ["log_m", "qz", "Rs"]:
            if p not in parameters.keys():
                parameters[p] = true_params[p]
        
        bulge = HernquistPotential(units,
                                   m=3.4E10*u.M_sun,
                                   c=0.7*u.kpc)
                                   
        disk = MiyamotoNagaiPotential(units,
                                      m=1.E11*u.M_sun, 
                                      a=6.5*u.kpc,
                                      b=0.26*u.kpc)
        halo = AxisymmetricNFWPotential(units,
                                        **parameters)
        
        super(Palomar5, self).__init__(units, 
                                       bulge=bulge,
                                       disk=disk,
                                       halo=halo)
        
        _params = halo._parameters.copy()
        _params.pop('r_0')
        
        self._acceleration_at = lambda r, n_particles, acc: pal5_acceleration(r, n_particles, acc, **_params)
        self._G = G.decompose(bases=units).value
        
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
        
        m_halo_enc = np.exp(self["halo"]._parameters["log_m"]) / 10.
        m_enc = self["disk"]._parameters["m"] + \
                self["bulge"]._parameters["m"] + \
                m_halo_enc
        
        return R_orbit * (m / (3.*m_enc))**(0.33333) #*25.
    
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

class Palomar5Logarithmic(CompositePotential):
    
    def __init__(self, **parameters):
        """ Represents the functional form of the Galaxy potential used by 
            Andreas' Palomar 5 simulation.
            
            Miyamoto-Nagai disk
            Hernquist bulge
            Axisymmetric, flattened NFW halo
            
            Model parameters: log_M, qz, Rs
            
            Parameters
            ----------
            parameters : dict
                A dictionary of parameters for the potential definition.
        """
        
        latex = ""
        
        units = (u.kpc, u.Myr, u.radian, u.M_sun)
        
        for p in ["v_c", "qz"]:
            if p not in parameters.keys():
                parameters[p] = true_params[p]
        
        halo = AxisymmetricLogarithmicPotential(units,
                                                **parameters)
        
        super(Palomar5Logarithmic, self).__init__(units, 
                                                  halo=halo)
        
        _params = halo._parameters.copy()
        _params.pop('r_0')
        
        #self._acceleration_at = lambda r, n_particles, acc: pal5_acceleration(r, n_particles, acc, **_params)
        self._G = G.decompose(bases=units).value
        
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
        
        m_halo_enc = self["halo"]._parameters["v_c"]**2 * R_orbit/self._G
        m_enc = m_halo_enc
        
        return R_orbit * (m / (3.*m_enc))**(0.33333) #*25.
    
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