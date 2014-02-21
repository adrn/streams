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

from .core import CartesianPotential, CompositePotential, PotentialParameter
from .common import MiyamotoNagaiPotential, HernquistPotential, LogarithmicPotentialLJ
from ._lm10_acceleration import lm10_acceleration
from .. import usys
from ..util import _parse_quantity

from ..inference.parameter import ModelParameter
from ..inference.prior import LogUniformPrior

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

        params = dict()
        params['q1'] = ModelParameter(name='q1', truth=1.38,
                            prior=LogUniformPrior(1.0,1.7))
        params['q2'] = ModelParameter(name='q2', truth=1.,
                            prior=LogUniformPrior(0.71,1.5))
        params['qz'] = ModelParameter(name='qz', truth=1.36,
                            prior=LogUniformPrior(1.0,1.7))
        params['phi'] = ModelParameter(name='phi',
                            truth=(97.*u.deg).decompose(usys).value,
                            prior=LogUniformPrior((80.*u.deg).decompose(usys).value,
                                                  (120.*u.deg).decompose(usys).value))
        params['v_halo'] = ModelParameter(name='v_halo',
                            truth=(121.858*u.km/u.s).decompose(usys).value,
                            prior=LogUniformPrior((100.*u.km/u.s).decompose(usys).value,
                                                  (200.*u.km/u.s).decompose(usys).value))
        params['R_halo'] = ModelParameter(name='R_halo',
                            truth=(12.*u.kpc).decompose(usys).value,
                            prior=LogUniformPrior((8.*u.kpc).decompose(usys).value,
                                                  (20*u.kpc).decompose(usys).value))

        self._parameter_dict = dict()
        self.parameters = dict(params)
        for k,v in self.parameters.items():
            #self.__dict__[k] = v
            self._parameter_dict[k] = parameters.get(k, params[k].truth)

        # bulge = HernquistPotential(usys,
        #                            m=3.4E10*u.M_sun,
        #                            c=0.7*u.kpc)

        # disk = MiyamotoNagaiPotential(usys,
        #                               m=1.E11*u.M_sun,
        #                               a=6.5*u.kpc,
        #                               b=0.26*u.kpc)

        # p_dict = dict([(k,v._value) for k,v in self.parameters.items()])
        # halo = LogarithmicPotentialLJ(usys, **p_dict)

        # super(LawMajewski2010, self).__init__(usys,
        #                                       bulge=bulge,
        #                                       disk=disk,
        #                                       halo=halo)
        self.units = usys
        self._G = G.decompose(bases=usys).value

    def _acceleration_at(self, r, n_particles, acc):
        return lm10_acceleration(r, n_particles, acc, **self._parameter_dict)

    def _enclosed_mass(self, R):
        """ Compute the enclosed mass at the position r. Assumes it's far from
            the disk and bulge, and is a spherically averaged approximation.
        """

        #m_halo_enc = self["halo"]._parameters["v_halo"]**2 * R/self._G
        #m_enc = self["disk"]._parameters["m"] + \
        #        self["bulge"]._parameters["m"] + \
        #        m_halo_enc

        # TODO: HACK!!!
        Rh = self._parameter_dict['R_halo']
        vh = self._parameter_dict['v_halo']
        G = self._G
        m_halo_enc = (2*R**3*vh**2) / (G*(R**2 + Rh**2))
        m_enc = 1.E11 + 3.4E10 + m_halo_enc

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

        Rh = self._parameter_dict['R_halo']
        dlnM_dlnR = (3*Rh**2 + R_orbit**2)/(Rh**2 + R_orbit**2)
        f = (1 - dlnM_dlnR/3.)**(-0.3333333333333)

        return R_orbit * (m / m_enc)**(0.3333333333333)

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

        r_unit = filter(lambda x: x.is_equivalent(u.km), self.units)[0]
        t_unit = filter(lambda x: x.is_equivalent(u.s), self.units)[0]
        return v_esc * r_unit/t_unit
