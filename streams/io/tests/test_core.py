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
from ..orphan import orphan_usys
from ...util import project_root

def test_add_sgr_coordinates():
    catalog = read_stripe82()
    
    catalog = add_sgr_coordinates(catalog)
    assert None not in catalog["Lambda"]
    assert None not in catalog["Beta"]
    assert None not in catalog["sgr_plane_dist"]
    
def test_read_table():
    data = read_table("ORP_SNAP", 
                      path=os.path.join(project_root, 'data', 'simulation', 'orphan'))

def test_table_to_particles():
    data = read_table("ORP_SNAP", 
                      path=os.path.join(project_root, 'data', 'simulation', 'orphan'))
    
    pc = table_to_particles(data, orphan_usys)

def test_table_to_orbits():
    data = read_table("ORP_CEN", 
                      path=os.path.join(project_root, 'data', 'simulation', 'orphan'))
    
    oc = table_to_orbits(data, orphan_usys)