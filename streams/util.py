# coding: utf-8

""" Utilities for the streams project. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import re
import logging
from datetime import datetime
import shutil

# Third-party
from astropy.utils.misc import isiterable
import astropy.units as u
import numpy as np

__all__ = ["_validate_coord", "project_root", "u_galactic"]

# Create logger
logger = logging.getLogger(__name__)

# This code will find the root directory of the project
_pattr = re.compile("(.*)\/streams")
try:
    matched_path = _pattr.search(os.getcwd()).groups()[0]
except AttributeError: # match not found, try __file__ instead
    matched_path = _pattr.search(__file__).groups()[0]

if os.path.basename(matched_path) == "streams":
    project_root = matched_path
else:
    project_root = os.path.join(matched_path, "streams")

#
def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])

u_galactic = [u.kpc, u.Myr, u.M_sun, u.radian]