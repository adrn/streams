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