# coding: utf-8

""" Utilities for the streams project. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import re

# Third-party
from astropy.utils.misc import isiterable
import astropy.units as u
import numpy as np

__all__ = ["_validate_coord", "project_root"]

_pattr = re.compile("(.*)\/streams")
project_root = _pattr.search(os.getcwd()).groups()[0]

def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])