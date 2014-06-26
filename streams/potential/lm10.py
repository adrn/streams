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
from scipy.signal import argrelmin,argrelmax

from .core import CartesianPotential, CompositePotential, PotentialParameter
from .common import MiyamotoNagaiPotential, HernquistPotential, LogarithmicPotentialLJ
from ._lm10_acceleration import lm10_acceleration, lm10_potential
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
                            prior=LogUniformPrior(1.1,1.8))
        params['q2'] = ModelParameter(name='q2', truth=1.,
                            prior=LogUniformPrior(0.71,1.5))
        params['qz'] = ModelParameter(name='qz', truth=1.36,
                            prior=LogUniformPrior(0.71,2.0))
        params['phi'] = ModelParameter(name='phi',
                            truth=(97.*u.deg).decompose(usys).value,
                            prior=LogUniformPrior((62.*u.deg).decompose(usys).value,
                                                  (132.*u.deg).decompose(usys).value))
        params['v_halo'] = ModelParameter(name='v_halo',
                            truth=(121.858*u.km/u.s).decompose(usys).value,
                            prior=LogUniformPrior((90.*u.km/u.s).decompose(usys).value,
                                                  (170.*u.km/u.s).decompose(usys).value))
        # params['R_halo'] = ModelParameter(name='R_halo',
        #                     truth=(12.*u.kpc).decompose(usys).value,
        #                     prior=LogUniformPrior((5.*u.kpc).decompose(usys).value,
        #                                           (40*u.kpc).decompose(usys).value))
        params['log_R_halo'] = ModelParameter(name='log_R_halo',
                            truth=np.log((12.*u.kpc).decompose(usys).value),
                            prior=LogUniformPrior(np.log((9.*u.kpc).decompose(usys).value),
                                                  np.log((25*u.kpc).decompose(usys).value)))

        self._parameter_dict = dict()
        self.parameters = dict(params)
        for k,v in self.parameters.items():
            #self.__dict__[k] = v
            self._parameter_dict[k] = parameters.get(k, params[k].truth)

        ####################
        # bulge = HernquistPotential(usys,
        #                            m=3.4E10*u.M_sun,
        #                            c=0.7*u.kpc)

        # disk = MiyamotoNagaiPotential(usys,
        #                               m=1.E11*u.M_sun,
        #                               a=6.5*u.kpc,
        #                               b=0.26*u.kpc)

        # p_dict = dict([(k,v) for k,v in self._parameter_dict.items()])
        # halo = LogarithmicPotentialLJ(usys, **p_dict)

        # super(LawMajewski2010, self).__init__(usys,
        #                                       bulge=bulge,
        #                                       disk=disk,
        #                                       halo=halo)
        ####################
        self.units = usys
        self._G = G.decompose(bases=usys).value
