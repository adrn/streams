# coding: utf-8

""" General utilities for reducing data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

# Create logger
logger = logging.getLogger(__name__)

def spectral_line_model(p, x):
    c, log_amplitude, stddev, mean = p
    return c + 10**log_amplitude / np.sqrt(2*np.pi*stddev**2.) * \
                np.exp(-0.5*(x-mean)**2/stddev**2)

def spetral_line_erf(p, x, y):
    return y - spectral_line_model(p, x)

def parse_wavelength(wvln, default_unit=u.angstrom):
    """ Parse a wavelength string from raw_input and return
        an astropy.units.Quantity object.
    """

    try:
        wvln_value, wvln_unit = wvln.split()
    except ValueError:
        # assume angstroms
        wvln_value = wvln
        wvln_unit = default_unit

    return float(wvln_value) * u.Unit(wvln_unit)