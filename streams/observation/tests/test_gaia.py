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

def test_gaia_spitzer_errors():

    from ...io import OrphanSimulation

    sim = OrphanSimulation()
    p = sim.particles(N=100, expr="tub!=0")
    with pytest.raises(ValueError):
        err = gaia_spitzer_errors(p)

    p = p.to_frame("heliocentric")
    err = gaia_spitzer_errors(p)