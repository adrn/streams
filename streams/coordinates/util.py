# coding: utf-8

""" Misc. utility functions """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

import astropy.units as u

__all__ = ["sex_to_dec", "dec_to_sex"]

def sex_to_dec(x, ms=False):
    """ Convert a sexagesimal representation to a decimal value.
        
        Parameters
        ----------
        x : tuple
            A length 3 tuple containing the components.
    """
    if ms:
        return x[0] + x[1]/60. + (x[2]+x[3]/1E6)/3600.
    else:
        return x[0] + x[1]/60. + x[2]/3600.

def dec_to_sex(x, ms=False):
    """ Convert a decimal value to a sexigesimal tuple.
        
        Parameters
        ----------
        x : numeric
    """
    a = int(x)
    _b = (x-a)*60.
    b = int(_b)
    
    if not ms:
        c = (_b - b)*60.
        return (a,b,c)
    else:
        _c = (_b - b)*60.
        c = int(_c)
        d = (_c-c)*1E6
        return (a,b,c,int(d))