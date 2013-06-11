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

from ...observation import read_stripe82, read_quest
from ..core import combine_catalogs, add_sgr_coordinates

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
    