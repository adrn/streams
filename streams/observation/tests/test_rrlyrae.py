# coding: utf-8
"""
    Test the RR Lyrae helper functions.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
import numpy as np
import pytest

from ..core import *
from ..rrlyrae import *

def test_distance():
    """ Test data from table 11 in:
        http://iopscience.iop.org/1538-3881/142/6/187/pdf/aj_142_6_187.pdf
    """
    test_data = [(-1.57, 15.62, 15.17), \
                 (-1.4, 12.15, 11.68), \
                 (-2.16, 15.51, 15.2), \
                 (-2.08, 15.51, 15.18)]
    
    for ii in range(len(test_data)):
        fe_h, m, dm = test_data[ii]
        pd = rrl_photometric_distance(m, fe_h)
        d = distance(dm).to(u.kpc)
        assert np.fabs(pd.to(u.kpc).value-d.to(u.kpc).value) < 0.1
        print(pd,d)