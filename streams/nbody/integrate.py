# coding: utf-8

""" Direct N-body """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["nbody_integrate"]

def nbody_integrate(particle_collection, time_steps, merge_length):
    """ TODO """
    