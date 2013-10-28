# coding: utf-8
"""
    Test observing classes
"""

from __future__ import absolute_import, unicode_literals, \
                       division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import pytest
from datetime import datetime

# Third-party
import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt

from ..observing import ObservingRun, ObservingNight

plot_path = "plots/tests/reduction"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_creation():

    path = os.path.join("/Users/adrian/Documents/GraduateSchool/Observing/",
                        "2013-10_MDM")
    obs_run = ObservingRun(path)

    utc = Time(datetime.utcnow(), scale="utc")
    night = ObservingNight(utc=utc, observing_run=obs_run)