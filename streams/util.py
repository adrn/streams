# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import re

# Third-party
from astropy.utils.misc import isiterable
from astropy.io import ascii
import astropy.units as u
import numpy as np

__all__ = ["_validate_coord", "pwd"]

pattr = re.compile("(.*)\/streams")
try:
    pwd = pattr.search(os.getcwd()).groups()[0]
except:
    pwd = ""

def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])