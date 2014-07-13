# coding: utf-8

""" Utilities.. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

# Project
from ..integrate import LeapfrogIntegrator

__all__ = ["log_normal"]

def log_normal(x, mu, sigma):
    X = x - mu
    return -0.5*(np.log(2*np.pi) + 2*np.log(sigma) + (X/sigma)**2)