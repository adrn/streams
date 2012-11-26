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

from ..core import Potential
from ..common import *

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

        potential.plot(np.logspace(7, 11, 100))
        plt.xscale("log")
        plt.yscale("symlog")
        #plt.show()

    def test_point_mass_3d(self):
        # Create a 3D cartesian point mass potential

        params = {"M" : 2E30, "location" : {"x0" : 0., "y0" : 0., "z0" : 0.}}
        potential = PointMassPotential(length_unit=u.au, time_unit=u.year, **params)
        assert potential.ndim == 3

        # Test with one coordinates
        c1 = potential.value_at(1., 0., 0.) # au

        # Test with individual coordinate arrays
        r_positions = np.array([[1., 0., 0.]])
        c2 = potential.value_at(r_positions)

        assert (c1 == c2).all()

        grid = np.linspace(-5, 5., 100)
        potential.plot(grid,grid,grid)
        plt.show()

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
        assert acc.shape == positions.shape

        assert (c1 == c2).all()

        with pytest.raises(ValueError):
            miya_potential.value_at(R_positions, z_positions[:-1])

        miya_potential.plot(np.linspace(-5., 5., 100), np.linspace(-5., 5., 100))
        plt.show()

    def test_composite_galaxy_3d(self):
        return
        G = 4.5E-12 # kpc^3 / M_sun / Myr

        def miyamoto_nagai(params): return lambda x,y,z: -G * params["M"] / np.sqrt(x**2 + y**2 + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)
        def d_miyamoto_nagai_dx(params): return lambda x,y,z: G * params["M"]*x / ((x**2 + y**2) + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5
        def d_miyamoto_nagai_dy(params): return lambda x,y,z: G * params["M"]*y / ((x**2 + y**2) + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5
        def d_miyamoto_nagai_dz(params): return lambda x,y,z: G * params["M"]*z*(1 + params["a"]/np.sqrt(z**2 + params["b"]**2)) / ((x**2 + y**2) + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5

        def log_halo(params): return lambda x,y,z: params["v_circ"]**2 / 2. * np.log(x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["d"]**2)
        def d_log_halo_dx(params): return lambda x,y,z: params["v_circ"]**2 * x / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["d"]**2)
        def d_log_halo_dy(params): return lambda x,y,z: params["v_circ"]**2 * y / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["d"]**2) / params["p"]**2
        def d_log_halo_dz(params): return  lambda x,y,z: params["v_circ"]**2 * z / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["d"]**2) / params["q"]**2

        disk_params = {"M" : 1E11, "a" : 6.5, "b" : 0.26}
        halo_params = {"v_circ" : (181.*u.km/u.s).to(u.kpc/u.Myr).value, "p" : 1., "q" : 1., "d" : 12.}

        potential = Potential()
        potential.add_component("disk", miyamoto_nagai(disk_params), derivs=(d_miyamoto_nagai_dx(disk_params), d_miyamoto_nagai_dy(disk_params), d_miyamoto_nagai_dz(disk_params)))
        potential.add_component("halo", log_halo(halo_params), derivs=(d_log_halo_dx(halo_params),d_log_halo_dy(halo_params),d_log_halo_dz(halo_params)))

        assert potential.ndim == 3

        # Test with two coordinates
        potential.value_at(6., 0., 1.1) # kpc

        # Test with one single coordinate array
        positions = np.array([[6., 0.1, 1.1], [8., 0.4, 10.], [24., 1., 11.2]]) # kpc
        c1 = potential.value_at(positions)

        # Test with individual coordinate arrays
        x_positions = np.array([6., 8., 24])
        y_positions = np.array([0.1, 0.4, 1.])
        z_positions = np.array([1.1, 10., 11.2]) # kpc
        c2 = potential.value_at(x_positions, y_positions, z_positions)

        acc = potential.acceleration_at(x_positions, y_positions, z_positions)
        assert acc.shape == positions.shape

        assert (c1 == c2).all()

        potential.plot(np.linspace(-5., 5., 100), np.linspace(-5., 5., 100), np.linspace(-5., 5., 100))
        #plt.show()

    def test_failure(self):

        potential = Potential()

        with pytest.raises(TypeError):
            potential.add_component("point mass", 15.)

        with pytest.raises(TypeError):
            potential.add_component("point mass blerg", lambda x: x, derivs=(5., ))

        with pytest.raises(ValueError):
            potential.add_component("point mass flerg", lambda x: x, derivs=(5., lambda x: x))

class TestPotentialAdd(object):

    def test_point_mass(self):
        params1 = {"M" : 1., "location" : {"x0" : -2., "y0" : 0., "z0" : 1.}}
        potential1 = PointMassPotential(length_unit=u.au, time_unit=u.year, **params1)
        params2 = {"M" : 1., "location" : {"x0" : 2., "y0" : 0., "z0" : 2.}}
        potential2 = PointMassPotential(length_unit=u.au, time_unit=u.year, **params2)

        new_potential = potential1 + potential2

        grid = np.linspace(-5, 5., 100)

        #otential1.plot(grid,grid,grid)
        #potential2.plot(grid,grid,grid)
        new_potential.plot(grid,grid,grid)
        plt.show()
        assert False

        #print(new_potential._potential_components)
        print(potential1.value_at(0.1, 0.1, 0.1))
        print(potential2.value_at(1.9, 1.9, 1.9))
        print(new_potential.value_at(0.1, 0.1, 0.1))
        print(new_potential.value_at(1.9, 1.9, 1.9))
        assert False

        # Test with one coordinates
        c1 = new_potential.value_at(1., 0., 0.) # au

        # Test with individual coordinate arrays
        r_positions = np.array([[1., 0., 0.]])
        c2 = new_potential.value_at(r_positions)

        assert (c1 == c2).all()

        grid = np.linspace(-5, 5., 100)
        new_potential.plot(grid,grid,grid)
        assert False
        plt.show()

    def test_composite_galaxy(self):
        # Define galaxy model parameters for potential

        disk_potential = MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
        bulge_potential = HernquistPotential(M=3.4E10*u.solMass, c=0.7)
        halo_potential = LogarithmicPotential(v_circ=(181.*u.km/u.s).to(u.kpc/u.Myr).value, p=1., q=0.75, c=12.)
        galaxy_potential = disk_potential + bulge_potential + halo_potential

        grid = np.linspace(-20, 20., 100)
        galaxy_potential.plot(grid,grid,grid)
        plt.show()
        assert False

        #print(new_potential._potential_components)
        print(potential1.value_at(0.1, 0.1, 0.1))
        print(potential2.value_at(1.9, 1.9, 1.9))
        print(new_potential.value_at(0.1, 0.1, 0.1))
        print(new_potential.value_at(1.9, 1.9, 1.9))
        assert False

        # Test with one coordinates
        c1 = new_potential.value_at(1., 0., 0.) # au

        # Test with individual coordinate arrays
        r_positions = np.array([[1., 0., 0.]])
        c2 = new_potential.value_at(r_positions)

        assert (c1 == c2).all()

        grid = np.linspace(-5, 5., 100)
        new_potential.plot(grid,grid,grid)
        assert False
        plt.show()
