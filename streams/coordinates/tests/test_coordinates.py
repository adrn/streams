# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import pytest
import numpy as np

from ...coordinates import Coordinates, CartesianCoordinates, SphericalCoordinates, CylindricalCoordinates

class TestCoordinates(object):
    def test1(self):
        # This should fail because Coordinates is just a base class
        with pytest.raises(TypeError):
            Coordinates(x=5.)

class TestCartesianCoordinates(object):

    def test_creation(self):
        # One dimensional cartesian coordinates
        c = CartesianCoordinates(x=15.)
        assert c.x == c._coords["x"]

        # Two dimensional cartesian coordinates
        c = CartesianCoordinates(x=15., y=11.)
        assert c.x == c._coords["x"]
        assert c.y == c._coords["y"]
        c = CartesianCoordinates(y=11., z=17.)

        # Three dimensional cartesian coordinates
        c = CartesianCoordinates(x=15., y=11., z=17.)
        assert c.x == c._coords["x"]
        assert c.y == c._coords["y"]
        assert c.z == c._coords["z"]

    def test_convert_spherical(self):
        # One dimensional cartesian coordinates
        c = CartesianCoordinates(x=15.).to(SphericalCoordinates)
        assert c.r == 15.
        assert not hasattr(c, "phi")
        assert not hasattr(c, "theta")

        # Two dimensional cartesian coordinates
        c = CartesianCoordinates(x=3., y=2.).to(SphericalCoordinates)
        assert c.r == np.sqrt(13.)
        np.testing.assert_almost_equal(c.phi, 0.588002604, 9)
        assert not hasattr(c, "theta")

        back_c = c.to(CartesianCoordinates)
        np.testing.assert_almost_equal(back_c.x, 3., 14)
        np.testing.assert_almost_equal(back_c.y, 2., 14)
        assert not hasattr(back_c, "z")

        c = CartesianCoordinates(y=11., z=17.).to(SphericalCoordinates)
        assert c.r == np.sqrt(410.)

        # Three dimensional cartesian coordinates
        c = CartesianCoordinates(x=15., y=11., z=17.).to(SphericalCoordinates)
        assert c.r == np.sqrt(635)
        np.testing.assert_almost_equal(c.phi,   0.63274883500, 9)
        np.testing.assert_almost_equal(c.theta, 0.83034054566, 9)

        back_c = c.to(CartesianCoordinates)
        np.testing.assert_almost_equal(back_c.x, 15, 14)
        np.testing.assert_almost_equal(back_c.y, 11, 14)
        np.testing.assert_almost_equal(back_c.z, 17, 14)

    def test_convert_cylindrical(self):
        # Two dimensional cartesian coordinates
        c = CartesianCoordinates(x=3., y=2.).to(CylindricalCoordinates)
        assert c.r == np.sqrt(13.)
        np.testing.assert_almost_equal(c.phi, 0.588002604, 9)
        assert not hasattr(c, "z")

        back_c = c.to(CartesianCoordinates)
        np.testing.assert_almost_equal(back_c.x, 3., 14)
        np.testing.assert_almost_equal(back_c.y, 2., 14)
        assert not hasattr(back_c, "z")

        c = CartesianCoordinates(y=11., z=17.).to(CylindricalCoordinates)
        assert c.r == 11.
        assert c.z == 17.
        assert not hasattr(c, "phi")

        c = CartesianCoordinates(x=41., z=17.).to(CylindricalCoordinates)
        assert c.r == 41.
        assert c.z == 17.
        assert not hasattr(c, "phi")

        # Three dimensional cartesian coordinates
        c = CartesianCoordinates(x=15., y=11., z=17.).to(CylindricalCoordinates)
        assert c.r == np.sqrt(346)
        np.testing.assert_almost_equal(c.phi, 0.63274883500, 9)
        assert c.z == 17.

        back_c = c.to(CartesianCoordinates)
        np.testing.assert_almost_equal(back_c.x, 15, 14)
        np.testing.assert_almost_equal(back_c.y, 11, 14)
        np.testing.assert_almost_equal(back_c.z, 17, 14)

class TestSphericalCoordinates(object):

    def test_creation(self):
        # One dimensional spherical coordinates
        c = SphericalCoordinates(r=15.)
        assert c.r == 15.

        # Two dimensional spherical coordinates
        c = SphericalCoordinates(r=15., phi=np.pi/2.)
        c = SphericalCoordinates(r=15., theta=np.pi/3.)

        # Three dimensional spherical coordinates
        c = SphericalCoordinates(r=15., theta=np.pi/4, phi=np.pi/6.)

    def test_convert_cylindrical(self):
        c = SphericalCoordinates(r=15., theta=0.8, phi=0.1).to(CylindricalCoordinates)
        np.testing.assert_almost_equal(c.r, 10.7603413634928, 12)
        assert c.phi == 0.1
        np.testing.assert_almost_equal(c.z, 10.4506006402074, 12)

class TestCylindricalCoordinates(object):

    def test_creation(self):
        # Two dimensional cylindrical coordinates
        c = CylindricalCoordinates(r=15., z=11.)

        # Three dimensional cylindrical coordinates
        c = CylindricalCoordinates(r=15., phi=np.pi/4., z=11.)
