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

__all__ = ["combine_catalogs", "add_sgr_coordinates", "radial_velocity"]

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

def radial_velocity(r, v):
    """ Compute the radial velocity in the heliocentric frame. """
    
    # the sun's velocity and position
    v_circ = 220.
    v_sun = np.array([0., v_circ, 0]) # km/s
    v_sun += np.array([9, 11., 6.]) # km/s
    r_sun = np.array([-8., 0, 0])
    
    # object's distance in relation to the sun(observed radius)
    r_rel = r - r_sun
    R_obs = np.sqrt(np.sum(r_rel**2, axis=-1))
    r_hat = r_rel / R_obs
    
    v_rel = v - v_sun
    v_hel = np.sqrt(np.sum((v_rel*r_hat)**2, axis=-1))
    
    return v_hel