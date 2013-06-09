# coding: utf-8
"""
    Test the core Potential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
from astropy.constants.si import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..lm10 import LawMajewski2010

def test_simple():
    p = LawMajewski2010()
    p = LawMajewski2010(v_halo=121*u.km/u.s)