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

from ..core import read_linear, read_quest
   
def test_linear():
    linear_data = read_linear()
    linear_data["ra"]
    linear_data["dec"]
    
def test_quest():
    quest_data = read_quest()
    quest_data["ra"]
    quest_data["dec"]