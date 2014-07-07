# coding: utf-8

""" Tools for wavelength calibration """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy.modeling import 

# For Hg-Ne lamps
line_list = np.array([5460.735, 5769.598, 5790.663, 5852.488, 5881.895, 
                      5944.834, 5975.534, 6029.997, 6074.338, 6096.163,
                      6128.450, 6143.062, 6163.594, 6217.281, 6266.495,
                      6304.789, 6334.428, 6382.992, 6402.246, 6506.528,
                      6532.882, 6598.953, 6678.276, 6717.043, 6929.467,
                      7032.413, 7173.938, 7245.167])

def scale_model(p, x):

def model(p, x):
    mus = line_list * scale_model()