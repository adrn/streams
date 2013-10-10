# coding: utf-8

""" Contains priors and likelihood functions for inferring parameters of
    the Logarithmic potential using back integration.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import math

# Third-party
import numpy as np
import astropy.units as u

from ..inference import generalized_variance
from ..potential.lm10 import LawMajewski2010, true_params, param_units

from .core import objective, objective2

__all__ = ["ln_posterior", "ln_likelihood", "objective"]
