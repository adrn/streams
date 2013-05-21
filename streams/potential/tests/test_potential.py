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
else:
    for plot in os.listdir(plot_path):
        os.remove(os.path.join(plot_path,plot))

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
    
    
    
def test_composite(self):
    
    potential = CompositePotential(units=[u.au,u.yr,u.M_sun], 
                               origin=[0.,0.,0.]*u.au)
    potential["one"] = PointMassPotential(units=potential.units,
                                          origin=[-1.,0.,0.]*u.au,
                                          m=1.*u.M_sun)
    print(potential.value_at([0.17157,0.,0.]*u.au), 
          potential.acceleration_at([0.17157,0.,0.]*u.au))
    
    potential["two"] = PointMassPotential(units=potential.units,
                                            origin=[1.,0.,0.]*u.au,
                                            m=0.5*u.M_sun)
    
    # Where forces cancel
    np.testing.assert_array_almost_equal(
                    sum(potential.acceleration_at([0.17157,0.,0.]*u.au)),
                    0.0, decimal=3)
    
    grid = np.linspace(-5.,5)*u.au
    fig,axes = potential.plot(grid,grid,grid)
    fig.savefig(os.path.join(plot_path, "two_point_mass.png"))

def test_many_point_masses(self, N=20):
    
    potential = CompositePotential(units=[u.au,u.yr,u.M_sun], 
                               origin=[0.,0.,0.]*u.au)
    
    for ii in range(N):
        org = np.random.uniform(-1., 1., size=3)
        org[2] = 0. # x-y plane
        potential[str(ii)] = PointMassPotential(units=potential.units,
                                          origin=org*u.au,
                                          m=np.random.uniform()*u.M_sun)
    
    grid = np.linspace(-1.,1,50)*u.au
    fig,axes = potential.plot(grid,grid,grid)
    fig.savefig(os.path.join(plot_path, "many_point_mass.png"))
        
        
        
# BELOW HERE NEEDS TO BE CLEANED

def test_failure():
    
    potential = CompositePotential(units=[u.kpc,u.Myr,u.M_sun],
                                   origin=[0.,0.,0.]*u.kpc)
    with pytest.raises(TypeError):
        potential["disk"] = "cat"
    
    with pytest.raises(TypeError):
        potential = CompositePotential(units=[u.kpc,u.Myr,u.M_sun],
                                   origin=[0.,0.,0.]*u.kpc,
                                   disk="cat")

def test_api():    
    # or, more complicated:
    mw_potential = CompositePotential(units=[u.kpc,u.Myr,u.M_sun,u.radian],
                                      origin=[0.,0.,0.]*u.kpc)
    mw_potential["disk"] = MiyamotoNagaiPotential(mw_potential.units,
                                                  m=1.E11*u.M_sun, 
                                                  a=6.5*u.kpc,
                                                  b=0.26*u.kpc,
                                                  origin=[0.,0.,0.]*u.kpc)
    
    mw_potential["bulge"] = HernquistPotential(mw_potential.units,
                                               m=3.4E10*u.M_sun,
                                               c=0.7*u.kpc,
                                               origin=[0.,0.,0.]*u.kpc)
            
    mw_potential["halo"] = LogarithmicPotentialLJ(mw_potential.units,
                                                  v_halo=(121.858*u.km/u.s),
                                                  q1=1.38,
                                                  q2=1.0,
                                                  qz=1.36,
                                                  phi=1.692969*u.radian,
                                                  r_halo=12.*u.kpc,
                                                  origin=[0.,0.,0.]*u.kpc)
    
    satellite_potential = CompositePotential(units=[u.kpc,u.Myr,u.M_sun], 
                                             origin=[40.,0.,0.]*u.kpc)
    satellite_potential["disk"] = MiyamotoNagaiPotential(mw_potential.units,
                                                         m=1.E11*u.M_sun, 
                                                         a=6.5*u.kpc,
                                                         b=0.26*u.kpc,
                                                         origin=[0.,0.,0.]*u.kpc)
    
    potential = CompositePotential(units=[u.kpc,u.Myr,u.M_sun], 
                                   origin=[0.,0.,0.]*u.kpc)
    
    potential["mw"] = mw_potential
    potential["satellite"] = satellite_potential

gal_units = [u.kpc, u.M_sun, u.Myr, u.radian]
class TestMiyamotoNagai(object):

    def test_creation(self):
        
        potential = MiyamotoNagaiPotential(units=gal_units, 
                                           m=1.E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai.png"))
    
    def test_composite(self):
        potential = CompositePotential(units=gal_units, 
                                   origin=[0.,0.,0.]*u.kpc)
        potential["disk"] = MiyamotoNagaiPotential(gal_units, 
                                           m=1E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)
        potential["imbh"] = PointMassPotential(gal_units, 
                                        m=2E9*u.M_sun, 
                                        origin=[5.,5.,0.]*u.kpc)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai_imbh.png"))

class TestHernquist(object):

    def test_creation(self):
        
        potential = HernquistPotential(units=gal_units, 
                                           m=1.E11*u.M_sun, 
                                           c=0.7*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "hernquist.png"))
    
    def test_composite(self):
        potential = CompositePotential(units=gal_units, 
                                   origin=[0.,0.,0.]*u.kpc)
        potential["disk"] = MiyamotoNagaiPotential(gal_units, 
                                           m=1E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)
        potential["bulge"] = HernquistPotential(units=gal_units, 
                                           m=1.E11*u.M_sun, 
                                           c=0.7*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)

        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "disk_bulge.png"))
        
class TestLogarithmicPotentialLJ(object):

    def test_creation(self):
        
        potential = LogarithmicPotentialLJ(units=gal_units, 
                                           q1=1.4,
                                           q2=1.,
                                           qz=1.5,
                                           phi=1.69*u.radian,
                                           v_halo=120.*u.km/u.s,
                                           r_halo=12.*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "log_halo_lj.png"))

class TestCompositeGalaxy(object):
    
    def test_creation(self):
        potential = CompositePotential(units=gal_units, 
                                       origin=[0.,0.,0.]*u.kpc)
        potential["disk"] = MiyamotoNagaiPotential(gal_units,
                                      m=1.E11*u.M_sun, 
                                      a=6.5*u.kpc,
                                      b=0.26*u.kpc,
                                      origin=[0.,0.,0.]*u.kpc)
        
        potential["bulge"] = HernquistPotential(gal_units,
                                   m=3.4E10*u.M_sun,
                                   c=0.7*u.kpc,
                                   origin=[0.,0.,0.]*u.kpc)
        
        potential["halo"] = LogarithmicPotentialLJ(gal_units,
                                      v_halo=(121.858*u.km/u.s),
                                      q1=1.38,
                                      q2=1.0,
                                      qz=1.36,
                                      phi=1.692969*u.radian,
                                      r_halo=12.*u.kpc,
                                      origin=[0.,0.,0.]*u.kpc)
                 
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "composite_galaxy.png"))