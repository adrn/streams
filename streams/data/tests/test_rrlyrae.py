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

from ..rrlyrae import *

@pytest.mark.parametrize(("reader", ), [(read_linear,),
                                        (read_quest,),
                                        (read_catalina,),
                                        (read_asas,),
                                        (read_nsvs,),
                                        (read_stripe82,)])
def test_readers(reader):
    data = reader()
    data["ra"]
    data["dec"]
    data["dist"]