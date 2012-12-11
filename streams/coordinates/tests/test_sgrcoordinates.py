# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import pytest
import numpy as np

import astropy.coordinates as coord
import astropy.units as u

from ..sgr import SgrCoordinates
from ...util import SGRData

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

    law_data = np.genfromtxt("streams/coordinates/tests/SgrCoord_data", names=True, delimiter=",")

    for row in law_data:
        c = coord.GalacticCoordinates(coord.Angle(row["l"], u.degree),
                                      coord.Angle(row["b"], u.degree))
        sgr_coords = c.transform_to(SgrCoordinates)
        print(sgr_coords.Lambda.degrees, row["lambda"])

    assert False

def test_with_simulation_data():
    import matplotlib.pyplot as plt

    sgr_data = SGRData(num_stars=10000)
    X = sgr_data.sgr_snap["x"] - 8. #kpc
    Y = sgr_data.sgr_snap["y"]
    Z = sgr_data.sgr_snap["z"]

    # Convert XYZ to Galactic latitude, longitude. Then convert to SgrCoordinates
    cps = [coord.CartesianPoints(x=X[ii], y=Y[ii], z=Z[ii], unit=u.kpc) for ii in range(len(X))]
    sgr_coords = [coord.GalacticCoordinates(cp).transform_to(SgrCoordinates) for cp in cps]

    lambdas = np.array([sgr.Lambda.degrees%360. for sgr in sgr_coords])
    betas = np.array([sgr.Beta.degrees for sgr in sgr_coords])

    lambdas -= 180.
    x = 2*np.sqrt(2)*np.cos(np.radians(betas))*np.sin(np.radians(lambdas/2.)) / np.sqrt(1 + np.cos(np.radians(betas))*np.cos(np.radians(lambdas/2.)))
    y = np.sqrt(2)*np.sin(np.radians(betas)) / np.sqrt(1 + np.cos(np.radians(betas))*np.cos(np.radians(lambdas/2.)))
    dists = np.array([sgr.distance.kpc for sgr in sgr_coords])

    # Plot in Galactic X-Y plane coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([sgr.x for sgr in sgr_coords], [sgr.y for sgr in sgr_coords], c='k', marker='.', s=1.)
    plt.show()

    # Plot in Galactic Spherical coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.degrees(x), np.degrees(y), c='k', marker='.', s=1.)
    plt.show()
