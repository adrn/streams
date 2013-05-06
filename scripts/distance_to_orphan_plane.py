# coding: utf-8

""" A function that computes the distance to the midplane of the Orphan orbit
    plane given an RA, Dec, and heliocentric distance.
    
    Note: This is the standalone script I sent to Branimir! Not meant for use
          in my own analysis.
    
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from numpy import radians, degrees, cos, sin
import astropy.coordinates as coord
from astropy.coordinates import OrphanCoordinates, distance_to_orphan_plane
import astropy.units as u
from astropy.io import ascii

def test_table():
    """ Test the script against table 2 values from Newberg et al. 2010 """
    names = ["Lambda", "l", "b", "db", "g0", "dg0", "v_gsr", "dv_gsr", "sigma_v", "N", "d", "dd"]
    table = """−30 173 46.5 0.7 18.8 0.2 115.5 6.7 11.5 4 46.8 4.5
    −20 187 50.0 1.0 18.5 0.1 119.7 6.9 11.9 4 40.7 1.9
    −9 205 52.5 0.7 18.0 0.1 139.8 4.6 12.9 9 32.4 1.5
    −1 218 53.5 1.0 17.8 0.1 131.5 3.1 8.2 8 29.5 1.4
    8 234 53.5 0.7 17.4 0.1 111.3 11.1 11.1 2 24.5 1.2
    18.4 249.5 50.0 0.7 17.1 0.1 101.4 2.9 9.8 12 21.4 1.0
    36 271 38.0 3.5 16.8 0.1 38.4 1.7 2.5 3 18.6 0.9"""
    
    t = ascii.read(table, names=names, data_start=0)
    
    for row in t:
        galactic = coord.GalacticCoordinates(row["l"], row["b"], unit=(u.degree, \
                                                                   u.degree))
        icrs = galactic.icrs
        print(distance_to_orphan_plane(icrs.ra, icrs.dec, float(row["d"])))
        
if __name__ == "__main__":
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Accepts an ra, dec, and "
        "heliocentric distance and returns distance to the Orphan stream orbit "
        "plane. Can alternatively specify Galacitc longitude and latitude "
        "instead.")
        
    parser.add_argument('--ra', type=str, dest='ra', default=None,
                       help='Right ascension in degrees.')
    parser.add_argument('--dec', type=str, dest='dec', default=None,
                       help='Declination in degrees.')
                       
    parser.add_argument('--l', type=str, dest='l', default=None,
                       help='Galactic longitude in degrees.')
    parser.add_argument('--b', type=str, dest='b', default=None,
                       help='Galactic latitude in degrees.')
                       
    parser.add_argument('--dist', type=str, dest='dist', 
                       help='Heliocentric distance in kpc.')
                       
    parser.add_argument('--test', dest='test', default=False, action="store_true",
                       help='Run tests.')
    
    args = parser.parse_args()
    
    if args.test:
        test_table()
        sys.exit(0)
    
    if args.dist == None:
        raise ValueError("dist is required!")
    
    if args.ra != None and args.dec != None:
        print(distance_to_orphan_plane(args.ra, args.dec, float(args.dist)))
    
    elif args.l != None and args.b != None:
        galactic = coord.GalacticCoordinates(args.l, args.b, unit=(u.degree, \
                                                                   u.degree))
        icrs = galactic.icrs
        print(distance_to_orphan_plane(icrs.ra, icrs.dec, float(args.dist)))
    
    else:
        raise ValueError("You must specify either RA/Dec or Galactic l/b")