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
               column_scales=dict(), path=None, N=None, expr=None):
    """ Read in data from an ASCII file. 
    
        Parameters
        ----------
        filename : str
        column_names : list (optional)
            If not specified, will try reading column_names from the data file.
        column_map : dict (optional)
            Rename column names from key -> value.
        column_scales : dict (optional)
            Multiply column name (key) by value.
        path : str (optional)
            Path to data file. If not specified, assumes <project_root>/data
        N : int (optional)
            Number of rows to return.
        expr : str (optional)
            Use numexpr to select out only rows that match criteria.
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
    
    if expr != None and len(expr.strip()) > 0:
        idx = numexpr.evaluate(str(expr), data)
        data = data[idx]
    
    if N != None and N > 0:
        idx = np.random.randint(0, len(data), N)
        data = data[idx]
    
    return data

def table_to_particles(table, unit_system, 
                       position_columns=None,
                       velocity_columns=None):
    """ Convert a astropy.table.Table-like object into a 
        ParticleCollection.
        
        Parameters
        ----------
        table : astropy.table.Table-like
        unit_system : streams.misc.UnitSystem
        position_columns : list
        velocity_columns_columns : list
    """
    
    r = np.zeros((len(data), 3))
    r[:,0] = np.array(data["x"])
    r[:,1] = np.array(data["y"])
    r[:,2] = np.array(data["z"])
    r = r*unit_system['length']
    
    v = np.zeros((len(data), 3))
    v[:,0] = np.array(data["vx"])
    v[:,1] = np.array(data["vy"])
    v[:,2] = np.array(data["vz"])
    v = v*unit_system['length']/unit_system['time']
    
    particles = ParticleCollection(r=r.to(u.kpc),
                                   v=v.to(u.km/u.s),
                                   m=np.zeros(len(r))*u.M_sun,
                                   units=unit_system)
    
    return particles

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
