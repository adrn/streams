# coding: utf-8

""" A function that computes the distance to the midplane of the Sagittarius orbit plane given an
    RA, Dec, and heliocentric distance.
    
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
from astropy.coordinates import SgrCoordinates
import astropy.units as u

def distance_to_sgr_plane(ra, dec, heliocentric_distance):
    """ Given an RA, Dec, and Heliocentric distance, compute the distance
        to the midplane of the Sgr plane (defined by Law & Majewski 2010).

        Parameters
        ----------
        ra : float
            A right ascension in decimal degrees
        dec : float
            A declination in decimal degrees
        heliocentric_distance : float
            The distance from the sun to a star in kpc.

    """

    eq_coords = coord.ICRSCoordinates(ra, dec, unit=(u.degree, u.degree))
    sgr_coords = eq_coords.transform_to(SgrCoordinates)
    sgr_coords.distance = coord.Distance(heliocentric_distance, unit=u.kpc)

    Z_sgr_sol = sgr_coords.distance.kpc * np.sin(sgr_coords.Beta.radians)

    return Z_sgr_sol
    
if __name__ == "__main__":
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Accepts an ra, dec, and "
        "heliocentric distance and returns distance to the Sgr plane. Can "
        "alternatively specify Galacitc longitude and latitude instead.")
        
    parser.add_argument('--ra', type=str, dest='ra', default=None,
                       help='Right ascension in degrees.')
    parser.add_argument('--dec', type=str, dest='dec', default=None,
                       help='Declination in degrees.')
                       
    parser.add_argument('--l', type=str, dest='l', default=None,
                       help='Galactic longitude in degrees.')
    parser.add_argument('--b', type=str, dest='b', default=None,
                       help='Galactic latitude in degrees.')
                       
    parser.add_argument('--dist', type=str, dest='dist', required=True,
                       help='Heliocentric distance in kpc.')
    
    args = parser.parse_args()
    
    if args.ra != None and args.dec != None:
        print(distance_to_sgr_plane(args.ra, args.dec, float(args.dist)))
    
    elif args.l != None and args.b != None:
        galactic = coord.GalacticCoordinates(args.l, args.b, unit=(u.degree, \
                                                                   u.degree))
        icrs = galactic.icrs
        print(distance_to_sgr_plane(icrs.ra, icrs.dec, float(args.dist)))
    
    else:
        raise ValueError("You must specify either RA/Dec or Galactic l/b")