# coding: utf-8
"""
    Test the coordinates class that represents the plane of orbit of the Sgr dwarf galaxy.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import pytest
import numpy as np

import astropy.coordinates as coord
import astropy.units as u

from ..orphan import OrphanCoordinates

def test_table():
    """ Test the transformation code against table 2 values from 
        Newberg et al. 2010 (below)
    """
    
    names = ["Lambda", "l", "b", "db", "g0", "dg0", "v_gsr", "dv_gsr", "sigma_v", "N", "d", "dd"]
    table = """-30 173 46.5 0.7 18.8 0.2 115.5 6.7 11.5 4 46.8 4.5
    -20 187 50.0 1.0 18.5 0.1 119.7 6.9 11.9 4 40.7 1.9
    -9 205 52.5 0.7 18.0 0.1 139.8 4.6 12.9 9 32.4 1.5
    -1 218 53.5 1.0 17.8 0.1 131.5 3.1 8.2 8 29.5 1.4
    8 234 53.5 0.7 17.4 0.1 111.3 11.1 11.1 2 24.5 1.2
    18.4 249.5 50.0 0.7 17.1 0.1 101.4 2.9 9.8 12 21.4 1.0
    36 271 38.0 3.5 16.8 0.1 38.4 1.7 2.5 3 18.6 0.9"""
    
    for line in table.split("\n"):
        Lambda,l,b = map(float, line.strip().split()[:3])
    
        galactic = coord.GalacticCoordinates(l, b, 
                                             unit=(u.degree, u.degree))
        orp = galactic.transform_to(OrphanCoordinates)
        print(orp.Lambda.format(unit=u.degree, decimal=True), 
              orp.Beta.format(unit=u.degree, decimal=True),
              Lambda)
            
        assert (np.fabs(orp.Lambda.degrees - Lambda) < 2.)