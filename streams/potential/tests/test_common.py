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
import matplotlib.pyplot as plt

from ..common import *

#with pytest.raises(TypeError):
#    quantity = 182.234 + u.meter

#np.testing.assert_array_almost_equal(new_quantity.value, 130.4164, decimal=5)

class TestMiyamotoNagaiPotential():

    def test_cartesian(self):
        potential = MiyamotoNagaiPotential(a=6.5, b=0.26, M=1E11*u.solMass)

        with pytest.raises(AttributeError):
            potential.add_component("halo", lambda x: x)

        grid = np.linspace(-10., 10., 100)
        potential.plot(grid, grid, grid)
        #plt.show()

        positions = np.array([[6., 0.1, 1.1], [8., 0.4, 10.], [24., 1., 11.2]]) # kpc
        #assert potential.acceleration_at(*positions[0]).shape == positions[0].shape
        #potential.acceleration_at(positions)

    def test_cylindrical(self):
        potential = MiyamotoNagaiPotential(a=6.5, b=0.26, M=1E11*u.solMass, coord_sys="cylindrical")

        with pytest.raises(AttributeError):
            potential.add_component("halo", lambda x: x)

        grid = np.linspace(-10., 10., 100)
        potential.plot(grid, np.zeros(len(grid)), grid)
        #plt.show()

class TestHernquistPotential():

    def test_cartesian(self):
        potential = HernquistPotential(c=0.7, M=3.4E10*u.solMass)

        with pytest.raises(AttributeError):
            potential.add_component("halo", lambda x: x)

        grid = np.linspace(-2., 2., 100)
        potential.plot(grid, grid, grid)
        #plt.show()

    def test_spherical(self):
        potential = HernquistPotential(c=0.7, M=3.4E10*u.solMass, coord_sys="spherical")

        with pytest.raises(AttributeError):
            potential.add_component("halo", lambda x: x)

        grid = np.linspace(0.05, 2., 100)
        phi_grid = np.zeros(len(grid))
        potential.plot(grid, phi_grid, grid)
        #plt.show()

class TestLogarithmicPotential():

    def test_cartesian(self):
        potential = LogarithmicPotential(p=0.95, q=0.75, c=12., v_circ=181.*u.km/u.s)

        with pytest.raises(AttributeError):
            potential.add_component("halo", lambda x: x)

        grid = np.linspace(-20., 20., 100)
        potential.plot(grid, grid, grid)
        #plt.show()

def time_miyamoto_derivative():
    from .._common import _miyamoto_nagai_dx
    import time

    params = dict()
    params["_G"] = 4.5E-12
    params["M"] = 1E11
    params["a"] = 6.5
    params["b"] = 0.26

    miyamoto_nagai_dx = lambda x,y,z: params["_G"] * params["M"]*x / ((x**2 + y**2) + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5

    # Many data points
    x = np.arange(1., 1000., int(1E6))
    y = np.arange(1., 1000., int(1E6))
    z = np.arange(1., 1000., int(1E6))

    time_1 = time.time()
    _miyamoto_nagai_dx(params["_G"],params["M"],params["a"],params["b"],x,y,z,len(x))
    print(time.time()-time_1)

    time_2 = time.time()
    miyamoto_nagai_dx(x,y,z)
    print(time.time()-time_2)

    # Smaller number of points
    x = np.arange(1., 1000., int(1E2))
    y = np.arange(1., 1000., int(1E2))
    z = np.arange(1., 1000., int(1E2))

    time_3 = time.time()
    _miyamoto_nagai_dx(params["_G"],params["M"],params["a"],params["b"],x,y,z,len(x))
    print(time.time()-time_1)

    time_4 = time.time()
    miyamoto_nagai_dx(x,y,z)
    print(time.time()-time_2)

    raise AttributeError
