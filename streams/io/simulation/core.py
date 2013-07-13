# coding: utf-8

""" Core reader function for ASCII files of simulation data. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.io.ascii as ascii

from ...util import project_root

__all__ = ["read_simulation"]

def read_simulation(filename, column_names, column_map=dict(), 
                    column_scales=dict(), path=None):
    """ Read in simulation data from an ASCII file. 
    
        Parameters
        ----------
        filename : str
        column_names : list
        column_map : dict (optional)
        path : str (optional)
    """
    
    if path is None:
        path = os.path.join(project_root, "data", "simulation")
    
    full_path = os.path.join(path, filename)
    
    # use astropy.io.ascii to read the ascii data
    data = ascii.read(full_path, names=column_names)
    
    # use the column map to rename columns
    for old_name,new_name in column_map.items():
        data.rename_column(old_name, new_name)
    
    # rescale columns using specified dict
    for name, scale in column_scales.items():
        data[name] = scale*data[name]
    
    return data