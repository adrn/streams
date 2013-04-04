# coding: utf-8
"""
    Test the helper classes for reading in project data.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

from ..core import LINEAR, QUEST
   
def test_coordinates():
    linear = LINEAR()
    quest = QUEST()
    
    assert isinstance(linear.ra[0], coord.RA)
    assert isinstance(linear.dec[0], coord.Dec)
    
    assert isinstance(quest.ra[0], coord.RA)
    assert isinstance(quest.dec[0], coord.Dec)
