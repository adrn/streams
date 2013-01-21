# coding: utf-8

""" Classes for accessing various data related to this project. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pyfits as pf

# Project
from ...util import project_root

__all__ = ["LM10", "LINEAR", "QUEST"]

def _make_npy_file(ascii_file, overwrite=False, ascii_kwargs=dict()):
    """ Make a .npy version of the given ascii file data. 
        
        Parameters
        ----------
        ascii_file : str
            The full path to the ascii file to convert.
        overwrite : bool (optional)
            If True, will overwrite any existing npy files and regenerate
            using the latest ascii data.
        ascii_kwargs : dict
            A dictionary of keyword arguments to be passed to ascii.read().
    """
    
    filename, ext = os.path.splitext(ascii_file)
    npy_filename = filename + ".npy"
    
    if os.path.exists(npy_filename) and overwrite:
        os.remove(npy_filename)
    elif os.path.exists(npy_filename) and not overwrite:
        return npy_filename
    
    data = ascii.read(dat_filename, **ascii_kwargs)
    np.save(npy_filename, np.array(data))
    return npy_filename

class LM10(object):
    
    def __init__(self, overwrite=False):
        """ Read in simulation data from Law & Majewski 2010. 
        
            Parameters
            ----------
            overwrite : bool (optional)
                If True, will overwrite any existing npy files and regenerate
                using the latest ascii data.
        """
        
        dat_filename = os.path.join(project_root, "data", "simulation", \
                                "SgrTriax_DYN.dat")
        
        npy_filename = _make_npy_file(dat_filename, overwrite=overwrite)
        self.data = np.load(npy_filename)

class LINEAR(object):
    
    def __init__(self, overwrite=False):
        """ Read in LINEAR RR Lyrae catalog from Branimir. """
        txt_filename = os.path.join(project_root, "data", "catalog", \
                                    "LINEAR_RRab.txt")
                                    
        npy_filename = _make_npy_file(txt_filename, overwrite=overwrite)
        self.data = np.load(npy_filename)

class QUEST(object):
    def __init__(self, overwrite=False):
        """ Read in QUEST RR Lyrae from Vivas et al. 2004 """
        fits_filename = os.path.join(project_root, "data", "catalog", \
                                    "quest_vivas2004_RRL.fits")
        
        hdulist = pf.open(fits_filename)
        self.data = hdulist[1].data
        