# coding: utf-8
"""
    Test the Potential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
from astropy.constants.si import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import *
from ..common import *

plot_path = "plots/tests/potential"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
#else:
#    for plot in os.listdir(plot_path):
#        os.remove(os.path.join(plot_path,plot))

class TestPointMass(object):
    usys = UnitSystem(u.au, u.M_sun, u.yr)
    
    def test_pointmass_creation(self):
        potential = PointMassPotential(unit_system=self.usys,
                                       m=1.*u.M_sun, 
                                       r_0=[0.,0.,0.]*u.au)
        
        # no mass provided
        with pytest.raises(AssertionError):
            potential = PointMassPotential(unit_system=self.usys, 
                                           r_0=[0.,0.,0.]*u.au)
        
        # no r_0 provided
        with pytest.raises(AssertionError):
            potential = PointMassPotential(unit_system=self.usys, 
                                           m=1.*u.M_sun)
    
    
    def test_pointmass_eval(self):
        potential = PointMassPotential(unit_system=self.usys,
                                       m=1.*u.M_sun, 
                                       r_0=[0.,0.,0.]*u.au)
                                       
        # Test with a single position
        r = [1.,0.,0.]*u.au
        pot_val = potential.value_at(r)
        assert pot_val.unit.is_equivalent(u.J/u.kg)
        _pot_val = potential._value_at(r.value)
        
        np.testing.assert_array_almost_equal(_pot_val, -39.487906, decimal=5) 
        
        acc_val = potential.acceleration_at(r)
        assert acc_val.unit.is_equivalent(u.m/u.s**2)
        _acc_val = potential._acceleration_at(r.value)
        
        np.testing.assert_array_almost_equal(_acc_val, 
                                             [-39.487906,0.,0.], decimal=5)
    
    def test_pointmass_plot(self):
        
        # 2-d case
        potential = PointMassPotential(unit_system=self.usys,
                                       m=1.*u.M_sun, 
                                       r_0=[0.,0.]*u.au)
        grid = np.linspace(-5.,5)*u.au
        fig,axes = potential.plot(ndim=2, grid=grid)
        fig.savefig(os.path.join(plot_path, "point_mass_2d.png"))
        
        # 3-d case
        potential = PointMassPotential(unit_system=self.usys,
                                       m=1.*u.M_sun, 
                                       r_0=[0.,0.,0.]*u.au)
        grid = np.linspace(-5.,5)*u.au
        fig,axes = potential.plot(ndim=3, grid=grid)
        fig.savefig(os.path.join(plot_path, "point_mass_3d.png"))

class TestComposite(object):
    usys = UnitSystem(u.au, u.M_sun, u.yr)
    
    def test_composite_create(self):
        potential = CompositePotential(unit_system=self.usys)
        
        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(unit_system=self.usys,
                                              m=1.*u.M_sun, 
                                              r_0=[0.,0.,0.]*u.au)
        
        # Try to add another with different units
        usys2 = UnitSystem(u.au, u.kg, u.s)
        with pytest.raises(TypeError):
            potential["two"] = PointMassPotential(unit_system=usys2,
                                                  m=1.*u.M_sun, 
                                                  r_0=[0.,0.,0.]*u.au)
        
        with pytest.raises(TypeError): 
            potential["two"] = "derp"

    def test_plot_composite(self):
        potential = CompositePotential(unit_system=self.usys)
        
        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(unit_system=self.usys,
                                              m=1.*u.M_sun, 
                                              r_0=[-1.,-1.,0.]*u.au)
        potential["two"] = PointMassPotential(unit_system=self.usys,
                                              m=1.*u.M_sun, 
                                              r_0=[1.,1.,0.]*u.au)      
        
        # Where forces cancel
        np.testing.assert_array_almost_equal(
                        potential.acceleration_at([0.,0.,0.]*u.au).value,
                        np.array([0.,0.,0.]), decimal=5)
        
        grid = np.linspace(-5.,5)*u.au
        fig,axes = potential.plot(ndim=3, grid=grid)
        fig.savefig(os.path.join(plot_path, "two_equal_point_masses.png"))
    
    def test_plot_composite2(self):
        potential = CompositePotential(unit_system=self.usys)
        
        # Add a point mass with same unit system
        potential["one"] = PointMassPotential(unit_system=self.usys,
                                              m=1.*u.M_sun, 
                                              r_0=[-1.,-1.,0.]*u.au)
        potential["two"] = PointMassPotential(unit_system=self.usys,
                                              m=5.*u.M_sun, 
                                              r_0=[1.,1.,0.]*u.au)
        
        grid = np.linspace(-5.,5)*u.au
        fig,axes = potential.plot(ndim=3, grid=grid)
        fig.savefig(os.path.join(plot_path, "two_different_point_masses.png"))

    def test_many_point_masses(self, N=20):
        potential = CompositePotential(unit_system=self.usys)
        
        for ii in range(N):
            r0 = np.random.uniform(-1., 1., size=3)
            r0[2] = 0. # x-y plane
            potential[str(ii)] = PointMassPotential(unit_system=self.usys,
                                                    m=np.random.uniform()*u.M_sun, 
                                                    r_0=r0*u.au)
        
        grid = np.linspace(-1.,1,50)*u.au
        fig,axes = potential.plot(ndim=3, grid=grid)
        fig.savefig(os.path.join(plot_path, "many_point_mass.png"))

class TestMiyamotoNagai(object):
    usys = UnitSystem(u.kpc, u.M_sun, u.Myr, u.radian)
    def test_miyamoto_creation(self):
        
        potential = MiyamotoNagaiPotential(unit_system=self.usys,
                                           m=1.E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           r_0=[0.,0.,0.]*u.kpc)
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(ndim=3, grid=grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai.png"))
    
    def test_composite(self):
        potential = CompositePotential(unit_system=self.usys)
        potential["disk"] = MiyamotoNagaiPotential(unit_system=self.usys,
                                           m=1.E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           r_0=[0.,0.,0.]*u.kpc)
        potential["imbh"] = PointMassPotential(unit_system=self.usys,
                                              m=2E9*u.M_sun, 
                                              r_0=[5.,5.,0.]*u.kpc)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(ndim=3, grid=grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai_imbh.png"))

class TestHernquist(object):
    usys = UnitSystem(u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):
        
        potential = HernquistPotential(unit_system=self.usys,
                                       m=1.E11*u.M_sun, 
                                       c=10.*u.kpc)
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
    
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid,ndim=3)
        fig.savefig(os.path.join(plot_path, "hernquist.png"))
        
class TestLogarithmicPotentialLJ(object):
    usys = UnitSystem(u.kpc, u.M_sun, u.Myr, u.radian)
    def test_create_plot(self):
        
        potential = LogarithmicPotentialLJ(unit_system=self.usys,
                                           q1=1.4,
                                           q2=1.,
                                           qz=1.5,
                                           phi=1.69*u.radian,
                                           v_halo=120.*u.km/u.s,
                                           r_halo=12.*u.kpc)
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid,ndim=3)
        fig.savefig(os.path.join(plot_path, "log_halo_lj.png"))

class TestCompositeGalaxy(object):
    usys = UnitSystem(u.kpc, u.M_sun, u.Myr, u.radian)
    def test_creation(self):
        potential = CompositePotential(unit_system=self.usys)
        potential["disk"] = MiyamotoNagaiPotential(unit_system=self.usys,
                                           m=1.E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           r_0=[0.,0.,0.]*u.kpc)
        
        potential["bulge"] = HernquistPotential(unit_system=self.usys,
                                       m=1.E11*u.M_sun, 
                                       c=0.7*u.kpc)
        
        potential["halo"] = LogarithmicPotentialLJ(unit_system=self.usys,
                                           q1=1.4,
                                           q2=1.,
                                           qz=1.5,
                                           phi=1.69*u.radian,
                                           v_halo=120.*u.km/u.s,
                                           r_halo=12.*u.kpc)
                 
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid=grid, ndim=3)
        fig.savefig(os.path.join(plot_path, "composite_galaxy.png"))
