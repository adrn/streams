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
                                             [-0.00593173285805,0.,0.], decimal=8)
    
    def test_addition(self):
        
        potential1 = PointMassPotential(m=1.*u.M_sun, 
                                        origin=[-2.,-1.,0.]*u.au)
        potential2 = PointMassPotential(m=1.*u.M_sun, 
                                        origin=[2.,1.,0.]*u.au)
        
        potential = potential1 + potential2
        
        grid = np.linspace(-5.,5)*u.au
        fig,axes = potential.plot(grid,grid,grid)
        fig.savefig(os.path.join(plot_path, "two_point_mass.png"))