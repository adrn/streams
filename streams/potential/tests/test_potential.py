# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import pytest
import numpy as np
from astropy.constants.si import G
import astropy.units as u

from ..core import Potential

#with pytest.raises(TypeError):
#    quantity = 182.234 + u.meter

#np.testing.assert_array_almost_equal(new_quantity.value, 130.4164, decimal=5)

class TestPotentialCreation():

    def test_point_mass_1d(self):
        # Create a 1-D point mass potential

        def point_mass(params):
            return lambda r: -G * params["M"] / r

        def d_point_mass_dr(params):
            return lambda r: G * params["M"] / r**2

        params = {"M" : 2E30} # kg
        potential = Potential()
        potential.add_component("point mass", point_mass(params), derivs=(d_point_mass_dr(params),))

        assert potential.ndim == 1

        # Test with one coordinates
        potential.value_at(6.E11) # m

        # Test with one single coordinate array
        positions = np.array([[6.E11], [8.E11], [24.E11]]) # kpc
        c1 = potential.value_at(positions)

        # Test with individual coordinate arrays
        r_positions = np.array([6.E11, 8E11, 24E11])
        c2 = potential.value_at(r_positions)

    def test_miyamoto_nagai_2d(self):
        G = 4.5E-12 # kpc^3 / M_sun / Myr

        def miyamoto_nagai(params):
            return lambda R, z: -G * params["M"] / np.sqrt(R**2 + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)

        def d_miyamoto_nagai_dR(params):
            return lambda R, z: G * params["M"]*R / (R**2 + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5

        def d_miyamoto_nagai_dz(params):
            return lambda R, z: G * params["M"]*z*(1 + params["a"]/np.sqrt(z**2 + params["b"]**2)) / (R**2 + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5

        params = {"M" : 1E11, "a" : 6.5, "b" : 0.1}
        miya_potential = Potential()
        miya_potential.add_component("miyamoto-nagai", miyamoto_nagai(params), derivs=(d_miyamoto_nagai_dR(params),d_miyamoto_nagai_dz(params)))

        assert miya_potential.ndim == 2

        # Test with two coordinates
        miya_potential.value_at(6., 0.1) # kpc

        with pytest.raises(ValueError):
            miya_potential.value_at(np.array([[1.,2,3], [4,5,6]]))

        # Test with one single coordinate array
        positions = np.array([[6., 0.1], [8., 0.4], [24., 1.]]) # kpc
        c1 = miya_potential.value_at(positions)

        # Test with individual coordinate arrays
        R_positions = np.array([6., 8., 24])
        z_positions = np.array([0.1, 0.4, 1.]) # kpc
        c2 = miya_potential.value_at(R_positions, z_positions)

        acc = miya_potential.acceleration_at(R_positions, z_positions)
        assert acc.shape == c2.shape

        assert (c1 == c2).all()

        with pytest.raises(ValueError):
            miya_potential.value_at(R_positions, z_positions[:-1])

    def test_failure(self):

        potential = Potential()

        with pytest.raises(TypeError):
            potential.add_component("point mass", 15.)

        with pytest.raises(TypeError):
            potential.add_component("point mass blerg", lambda x: x, derivs=(5., ))

        with pytest.raises(ValueError):
            potential.add_component("point mass flerg", lambda x: x, derivs=(5., lambda x: x))

