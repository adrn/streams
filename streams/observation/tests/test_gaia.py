# coding: utf-8
"""
    Test the GAIA observational error estimate functions.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import pytest

from ..gaia import *
from ...data.sgr import lm10_particles

plot_path = "plots/tests/observation"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    
def test_against_table():
    """ Test astrometric precision against the table at:
        http://www.rssd.esa.int/index.php?project=GAIA&page=Science_Performance
    """
    
    # B1V star
    np.testing.assert_almost_equal(parallax_error(15, -0.22).value, 26E-6, 5)
    np.testing.assert_almost_equal(parallax_error(20, -0.22).value, 330E-6, 4)
    
    # G2V star
    np.testing.assert_almost_equal(parallax_error(15, 0.75).value, 24E-6, 6)
    np.testing.assert_almost_equal(parallax_error(20, 0.75).value, 290E-6, 4)
    
    # M6V star
    np.testing.assert_almost_equal(parallax_error(15, 3.85).value, 9E-6, 4)
    np.testing.assert_almost_equal(parallax_error(20, 3.85).value, 100E-6, 4)

def test_add_uncertainties():
    
    N = 20
    x = np.random.uniform(0., 10., (3,N)) * u.kpc
    v = np.random.uniform(0., 25., (3,N)) * u.km/u.s
    
    derp = rr_lyrae_add_observational_uncertainties(x[0],x[1],x[2],v[0],v[1],v[2])

def test_particles_uncertainties():
    #particles = add_uncertainties_to_particles(particles, distance_error_percent, radial_velocity_error)
    
    particles = lm10_particles(N=100)
    
    fig,axes = particles.plot_r('xyz')
    particles_error = add_uncertainties_to_particles(particles, distance_error_percent=2)
    fig,axes = particles_error.plot_r('xyz', axes=axes, scatter_kwargs=dict(color='r'))
    fig.savefig(os.path.join(plot_path, "particle_uncertainties_r_2percent.png"))
    
    fig,axes = particles.plot_r('xyz')
    particles_error = add_uncertainties_to_particles(particles, distance_error_percent=20.)
    fig,axes = particles_error.plot_r('xyz', axes=axes, scatter_kwargs=dict(color='r'))
    fig.savefig(os.path.join(plot_path, "particle_uncertainties_r_20percent.png"))
    
    fig,axes = particles.plot_v(['vx','vy','vz'])
    particles_error = add_uncertainties_to_particles(particles, 
                                                     radial_velocity_error=10*u.km/u.s)
    fig,axes = particles_error.plot_v(['vx','vy','vz'], axes=axes, 
                                      scatter_kwargs=dict(color='r'))
    fig.savefig(os.path.join(plot_path, "particle_uncertainties_RV_10kms.png"))
    
    fig,axes = particles.plot_v(['vx','vy','vz'])
    particles_error = add_uncertainties_to_particles(particles, 
                                                     radial_velocity_error=20*u.km/u.s)
    fig,axes = particles_error.plot_v(['vx','vy','vz'], axes=axes, 
                                      scatter_kwargs=dict(color='r'))
    fig.savefig(os.path.join(plot_path, "particle_uncertainties_RV_20kms.png"))