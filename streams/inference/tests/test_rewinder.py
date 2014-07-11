# coding: utf-8

""" Test the Rewinder model subclass """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from ..rewinder import Rewinder

logger.setLevel(logging.DEBUG)

def test_config():
    rw = Rewinder.from_config("/Users/adrian/projects/streams/config/test.yml")
