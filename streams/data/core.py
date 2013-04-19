# coding: utf-8

""" Classes for accessing various data related to this project. """

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.coordinates as coord
import astropy.units as u

# Project
from ..util import project_root

__all__ = ["read_linear", "read_quest"]

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

    data = ascii.read(ascii_file, **ascii_kwargs)
    np.save(npy_filename, np.array(data))
    return npy_filename

def read_linear():
    """ Read in the LINEAR data -- RR Lyrae from the LINEAR survey, 
        sent to me from Branimir. 
    """
    txt_filename = os.path.join(project_root, "data", "catalog", \
                                "LINEAR_RRab.txt")
    data = ascii.read(txt_filename)    
    return data

def read_quest():
    """ Read in the QUEST data -- RR Lyrae from the QUEST survey,
        Vivas et al. 2004. 
        
        - Covers Stripe 82
    """
    fits_filename = os.path.join(project_root, "data", "catalog", \
                                "quest_vivas2004_RRL.fits")
    hdulist = fits.open(fits_filename)
    tb_data = np.array(hdulist[1].data)
    data = Table(tb_data)
    
    # Map RA/Dec columns to easier names
    data.rename_column('RAJ2000', str("ra"))
    data.rename_column('DEJ2000', str("dec"))
    
    return data
