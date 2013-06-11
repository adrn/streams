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

def test_combine_catalogs():
    stripe82 = read_stripe82()
    quest = read_quest()
    
    catalog = combine_catalogs(Stripe82=stripe82, QUEST=quest)
    assert "Stripe82" in catalog["survey"] and "QUEST" in catalog["survey"]
    
    # make sure there are no null values in ra, dec, dist, and survey
    assert None not in catalog["survey"]
    assert None not in catalog["ra"]
    assert None not in catalog["dec"]
    assert None not in catalog["dist"]

def test_add_sgr_coordinates():
    stripe82 = read_stripe82()
    quest = read_quest()
    catalog = combine_catalogs(Stripe82=stripe82, QUEST=quest)
    
    catalog = add_sgr_coordinates(catalog)
    assert None not in catalog["Lambda"]
    assert None not in catalog["Beta"]
    assert None not in catalog["sgr_plane_dist"]
    
def test_radial_velocity():
    
    r = np.array([-8., 1., 0.])
    v = np.array([0., -220., 0.])
    assert radial_velocity(r, v) == 451.
    
    r = np.array([-8., -1., 0.])
    v = np.array([0., 220., 0.])
    assert radial_velocity(r, v) == 11.