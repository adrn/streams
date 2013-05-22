# coding: utf-8
""" Test the special fucntions used to compute our scalar objective """

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import numpy as np
import pytest
import astropy.units as u

from ..potential.lm10 import CLawMajewski2010
from ..nbody import Particle

def test_r_tide():
    
    potential = CLawMajewski2010()
    satellite = Particle()