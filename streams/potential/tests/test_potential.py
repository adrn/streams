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

def test_api():
    # API:
    potential = CompositePotential(units=[u.au,u.yr,u.M_sun], 
                                   origin=[0.,0.,0.]*u.au)
    potential["sun"] = PointMassPotential(units=potential.units,
                                          origin=[0.,0.,0.]*u.au,
                                          m=1.*u.M_sun)
    potential["earth"] = PointMassPotential(units=potential.units,
                                            origin=[1.,0.,0.]*u.au,
                                            m=3E-6*u.M_sun)
    
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
        potential = PointMassPotential(unit_bases=[u.au, u.M_sun, u.yr],
                                       m=1.*u.M_sun, 
                                       origin=[0.,0.,0.]*u.au)
        
        r = [1.,0.,0.]*u.au
        pot_val = potential.value_at(r)
        
        np.testing.assert_array_almost_equal(
            np.sqrt(-pot_val), 2*np.pi, decimal=3) 
        
        acc_val = potential.acceleration_at(r)
        
        np.testing.assert_array_almost_equal(acc_val, 
                                             [-39.48621,0.,0.], decimal=2)
    
    def test_addition(self):
        
        bases = [u.au, u.M_sun, u.yr]
        potential1 = PointMassPotential(bases, 
                                        m=1.*u.M_sun, 
                                        origin=[-2.,-1.,0.]*u.au)
        potential2 = PointMassPotential(bases,
                                        m=1.*u.M_sun, 
                                        origin=[2.,1.,0.]*u.au)
        
        potential = potential1 + potential2
        
        grid = np.linspace(-5.,5)*u.au
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "two_point_mass.png"))

gal_bases = [u.kpc, u.M_sun, u.Myr, u.radian]
class TestMiyamotoNagai(object):

    def test_creation(self):
        
        potential = MiyamotoNagaiPotential(gal_bases, 
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
    
    def test_addition(self):
        
        potential1 = MiyamotoNagaiPotential(gal_bases, 
                                           m=1E11*u.M_sun, 
                                           a=6.5*u.kpc,
                                           b=0.26*u.kpc,
                                           origin=[0.,0.,0.]*u.kpc)
        potential2 = PointMassPotential(gal_bases, 
                                        m=2E9*u.M_sun, 
                                        origin=[12.,1.,0.]*u.kpc)
        potential = potential1 + potential2
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "miyamoto_nagai_smbh.png"))

class TestCompositeGalaxy(object):
    
    def test_creation(self):

        disk = MiyamotoNagaiPotential(gal_bases,
                                      m=1.E11*u.M_sun, 
                                      a=6.5*u.kpc,
                                      b=0.26*u.kpc,
                                      origin=[0.,0.,0.]*u.kpc)
        
        bulge = HernquistPotential(gal_bases,
                                   m=3.4E10*u.M_sun,
                                   c=0.7*u.kpc,
                                   origin=[0.,0.,0.]*u.kpc)
        
        halo = LogarithmicPotentialLJ(gal_bases,
                                      v_halo=(121.858*u.km/u.s),
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
            acc_val = potential.acceleration_at(r)
        
        potential = disk + bulge + halo 
        
        r = [1.,0.,0.]*u.kpc
        pot_val = potential.value_at(r)        
        acc_val = potential.acceleration_at(r)
        
        grid = np.linspace(-20.,20, 50)*u.kpc
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "composite_galaxy.png"))