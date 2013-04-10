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

#np.testing.assert_array_almost_equal(new_quantity.value, 130.4164, decimal=5)

plot_path = "plots/tests/potential"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)
else:
    for plot in os.listdir(plot_path):
        os.remove(os.path.join(plot_path,plot))

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
    mw_potential = CompositePotential(units=[u.kpc,u.Myr,u.M_sun],
                                      origin=[0.,0.,0.]*u.kpc)
    mw_potential["disk"] = MiyamotoNagaiPotential(galaxy_potential.units,
                                                  m=1.E11*u.M_sun, 
                                                  a=6.5*u.kpc,
                                                  b=0.26*u.kpc,
                                                  origin=[0.,0.,0.]*u.kpc)
    
    mw_potential["bulge"] = HernquistPotential(galaxy_potential.units,
                                               m=3.4E10*u.M_sun,
                                               c=0.7*u.kpc,
                                               origin=[0.,0.,0.]*u.kpc)
            
    mw_potential["halo"] = LogarithmicPotentialLJ(galaxy_potential.units,
                                                  v_halo=(121.858*u.km/u.s),
                                                  q1=1.38,
                                                  q2=1.0,
                                                  qz=1.36,
                                                  phi=1.692969*u.radian,
                                                  r_halo=12.*u.kpc,
                                                  origin=[0.,0.,0.]*u.kpc)
    
    satellite_potential = CompositePotential(units=[u.kpc,u.Myr,u.M_sun], 
                                             origin=[40.,0.,0.]*u.kpc)
    satellite_potential["disk"] = MiyamotoNagaiPotential(galaxy_potential.units,
                                                         m=1.E11*u.M_sun, 
                                                         a=6.5*u.kpc,
                                                         b=0.26*u.kpc,
                                                         origin=[0.,0.,0.]*u.kpc)
    
    potential = CompositePotential(units=[u.kpc,u.Myr,u.M_sun], 
                                   origin=[0.,0.,0.]*u.kpc)
    
    potential["mw"] = mw_potential
    potential["satellite"] = satellite_potential

class TestPointMass(object):

    def test_creation(self):
        potential = PointMassPotential(units=[u.au, u.M_sun, u.yr],
                                       m=1.*u.M_sun, 
                                       origin=[0.,0.,0.]*u.au)
        
        with pytest.raises(AssertionError):
            potential = PointMassPotential(units=[u.au, u.M_sun, u.yr],
                                       origin=[0.,0.,0.]*u.au)
        
        r = [1.,0.,0.]*u.au
        pot_val = potential.value_at(r)
        
        np.testing.assert_array_almost_equal(
            np.sqrt(-pot_val), 2*np.pi, decimal=3) 
        
        acc_val = potential.acceleration_at(r)
        
        np.testing.assert_array_almost_equal(acc_val, 
                                             [-39.48621,0.,0.], decimal=2)
        
        grid = np.linspace(-5.,5)*u.au
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "one_point_mass.png"))
    
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