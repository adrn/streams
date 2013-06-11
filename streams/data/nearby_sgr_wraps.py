# coding: utf-8

""" Code for helping to select stars from the nearby Sgr wraps. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import numexpr
import astropy.units as u
from astropy.table import vstack, Table, Column

# Project

__all__ = []

#lm10 = lm10_particles(expr="(Pcol>-1) & (Pcol<6) & (abs(Lmflag)==1)")

def combine_catalogs(**kwargs):
    """ Combine multiple catalogs of data into the same Table.
    
        Parameters
        ----------
        kwargs
            Key's should be the names of the catalogs, values should be
            the data itself in the form of astropy.table.Table objects.
            They should all have ra, dec, and dist columns.
    """
    
    for name,data in kwargs.items():
        c = Column([name]*len(data), name='survey')
        data.add_column(c)
    
    data = vstack(kwargs.values())
    
    return data