# coding: utf-8
"""
    Test the helper classes for reading in Sgr data.
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

from .. import lm10_particles, lm10_satellite, lm10_time

def test_inits():
    p = lm10_particles()
    s = lm10_satellite()
    t1,t2 = lm10_time()