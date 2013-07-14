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
import astropy.coordinates as coord

# Project
from ..coordinates import SgrCoordinates, distance_to_sgr_plane

__all__ = ["read_table", "add_sgr_coordinates"]

def read_table(filename, column_names=None, column_map=dict(), 
               column_scales=dict(), path=None):
    """ Read in data from an ASCII file. 
    
        Parameters
        ----------
        filename : str
        column_names : list (optional)
            If not specified, will try reading column_names from the data file.
        column_map : dict (optional)
        path : str (optional)
    """
    
    if path is None:
        path = os.path.join(project_root, "data")
    
    full_path = os.path.join(path, filename)
    
    # use astropy.io.ascii to read the ascii data
    data = ascii.read(full_path, names=column_names)
    
    if column_names is None and data.colnames[0].contains('col'):
        raise IOError("Failed to read column names from file.")
    
    # use the column map to rename columns
    for old_name,new_name in column_map.items():
        data.rename_column(old_name, new_name)
    
    # rescale columns using specified dict
    for name, scale in column_scales.items():
        data[name] = scale*data[name]
    
    return data

def table_to_particles(table, column_map=None):
    """ Convert a astropy.table.Table-like object into a 
        ParticleCollection.
        
        Parameters
        ----------
        table : astropy.table.Table-like
        column_map : dict (optional)
    """
    # TODO
    pass

def add_sgr_coordinates(data):
    """ Given a table of catalog data, add columns with Sagittarius 
        Lambda and Beta coordinates.
        
        Parameters
        ----------
        data : astropy.table.Table
            Must contain ra, dec, and dist columns.
    """
    
    L,B = [], []
    sgr_plane_dist = []
    for star in data:
        sgr_plane_dist.append(distance_to_sgr_plane(star['ra'],
                                                    star['dec'],
                                                    star['dist']))
        icrs = coord.ICRSCoordinates(star['ra'],star['dec'],unit=(u.deg,u.deg))
        sgr = icrs.transform_to(SgrCoordinates)
        L.append(sgr.Lambda.degrees)
        B.append(sgr.Beta.degrees)
    
    Lambda = Column(L, name='Lambda', units=u.degree)
    Beta = Column(B, name='Beta', units=u.degree)
    sgr_plane_D = Column(sgr_plane_dist, name='sgr_plane_dist', units=u.kpc)
    data.add_column(Lambda)
    data.add_column(Beta)
    data.add_column(sgr_plane_D)
    
    return data
