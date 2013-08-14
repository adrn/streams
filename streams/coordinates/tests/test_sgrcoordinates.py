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

from ..sgr import SgrCoordinates

def test_simple():
    c = coord.ICRSCoordinates(coord.Angle(217.2141, u.degree),
                              coord.Angle(-11.4351, u.degree))
    c.transform_to(SgrCoordinates)

    c = coord.GalacticCoordinates(coord.Angle(217.2141, u.degree),
                                  coord.Angle(-11.4351, u.degree))
    c.transform_to(SgrCoordinates)

    c = SgrCoordinates(coord.Angle(217.2141, u.degree),
                       coord.Angle(-11.4351, u.degree))
    c.transform_to(coord.ICRSCoordinates)
    c.transform_to(coord.GalacticCoordinates)

def test_against_David_Law():
    """ Test my code against an output file from using David Law's cpp code. Do:

            g++ SgrCoord.cpp; ./a.out

        to generate the data file, SgrCoord_data.

    """

    law_data = np.genfromtxt("streams/coordinates/tests/SgrCoord_data", 
                             names=True, delimiter=",")

    for row in law_data:
        c = coord.GalacticCoordinates(coord.Angle(row["l"], u.degree),
                                      coord.Angle(row["b"], u.degree))
        sgr_coords = c.transform_to(SgrCoordinates)
        law_sgr_coords = SgrCoordinates(row["Lambda"], row["Beta"], 
                                        unit=(u.degree, u.degree))
        
        print(sgr_coords, law_sgr_coords)
        
        sep = sgr_coords.separation(law_sgr_coords).arcsecs*u.arcsec
        print(sep)
        assert sep < 1.*u.arcsec
