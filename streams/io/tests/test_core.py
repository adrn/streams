# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest

from ..catalogs import read_stripe82, read_quest
from ..core import *

def test_add_sgr_coordinates():
    stripe82 = read_stripe82()
    quest = read_quest()
    catalog = combine_catalogs(Stripe82=stripe82, QUEST=quest)
    
    catalog = add_sgr_coordinates(catalog)
    assert None not in catalog["Lambda"]
    assert None not in catalog["Beta"]
    assert None not in catalog["sgr_plane_dist"]
    
def test_radial_velocity():
    
    r1 = np.array([-8., 1., 0.])
    v1 = np.array([0., -220., 0.])
    assert radial_velocity(r1, v1) == -451.
    
    r2 = np.array([-8., -1., 0.])
    v2 = np.array([0., 220., 0.])
    assert radial_velocity(r2, v2) == 11.
    
    r = np.vstack((r1, r2))
    v = np.vstack((v1, v2))
    assert (radial_velocity(r, v) == np.array([-451, 11.])).all()