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

from ..core import CartesianPotential
from ..common import *

#np.testing.assert_array_almost_equal(new_quantity.value, 130.4164, decimal=5)

plot_path = "plots/tests/potential"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)
else:
    for plot in os.listdir(plot_path):
        os.remove(os.path.join(plot_path,plot))

class TestPointMass(object):

    def test_creation(self):
        potential = PointMassPotential(m=1.*u.M_sun, 
                                       origin=[0.,0.,0.]*u.au)
        
        r = [1.,0.,0.]*u.au
        pot_val = potential.value_at(r)
        assert pot_val.unit == (u.au/u.s)**2
        
        np.testing.assert_array_almost_equal(
            np.sqrt(-pot_val.decompose(bases=[u.au,u.yr])), 2*np.pi, decimal=3) 
        
        acc_val = potential.acceleration_at(r)
        assert acc_val.unit.is_equivalent(u.m/u.s**2)
        
        np.testing.assert_array_almost_equal(acc_val.to(u.m/u.s**2).value, 
                                             [0.00593173285805,0.,0.], decimal=8)
    
    def test_addition(self):
        
        potential1 = PointMassPotential(m=1.*u.M_sun, 
                                        origin=[-2.,-1.,0.]*u.au)
        potential2 = PointMassPotential(m=1.*u.M_sun, 
                                        origin=[2.,1.,0.]*u.au)
        
        potential = potential1 + potential2
        
        grid = np.linspace(-5.,5)*u.au
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "two_point_mass.png"))

class TestMiyamotoNagai(object):

    def test_creation(self):

        potential = MiyamotoNagaiPotential(m=1.E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)

        assert pot_val.unit.is_equivalent((u.kpc/u.s)**2)
        
        acc_val = potential.acceleration_at(r)
        assert acc_val.unit.is_equivalent(u.m/u.s**2)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai.png"))
    
    def test_addition(self):
        
        potential1 = MiyamotoNagaiPotential(m=1E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)
        potential2 = PointMassPotential(m=2E9*u.M_sun, 
                                        origin=[12.,1.,0.]*u.kpc)
        potential = potential1 + potential2
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai_smbh.png"))

class TestCompositeGalaxy(object):
    
    def test_creation(self):

        disk = MiyamotoNagaiPotential(m=1.E11*u.M_sun, 
                                      a=6.5*u.kpc,
                                      b=0.26*u.kpc,
                                      origin=[0.,0.,0.]*u.kpc)
        
        bulge = HernquistPotential(m=3.4E10*u.M_sun,
                                   c=0.7*u.kpc,
                                   origin=[0.,0.,0.]*u.kpc)
        
        halo = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s),
                                      q1=1.38,
                                      q2=1.0,
                                      qz=1.36,
                                      phi=1.692969*u.radian,
                                      r_halo=12.*u.kpc,
                                      origin=[0.,0.,0.]*u.kpc)
        
        r = [1.,0.,0.]*u.kpc
        for potential in [disk, bulge, halo]: 
            print(potential)
            pot_val = potential.value_at(r)
            assert pot_val.unit.is_equivalent((u.kpc/u.s)**2)        
            
            acc_val = potential.acceleration_at(r)
            assert acc_val.unit.is_equivalent(u.m/u.s**2)
        
        potential = disk + bulge + halo 
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)

        assert pot_val.unit.is_equivalent((u.kpc/u.s)**2)
        
        acc_val = potential.acceleration_at(r)
        assert acc_val.unit.is_equivalent(u.m/u.s**2)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "composite_galaxy.png"))