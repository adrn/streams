# coding: utf-8

""" Classes for accessing various data related to this project. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

# Project
from ...util import project_root

class LM10(object):
    
    def __init__(self):
        """ Read in simulation data from Law & Majewski 2010. """
        filename = os.path.join(project_root, "data", "simulation", \
                                "SgrTriax_DYN.npy")
        
        self.data = np.load(filename)