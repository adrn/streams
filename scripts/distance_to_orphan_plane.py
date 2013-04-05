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
from astropy.coordinates import transformations
from astropy.coordinates.angles import rotation_matrix
import astropy.units as u
from astropy.io import ascii

class OrphanCoordinates(coord.SphericalCoordinatesBase):
    """ A spherical coordinate system defined by the orbit of the Orphan stream
        as described in 
        http://iopscience.iop.org/0004-637X/711/1/32/pdf/apj_711_1_32.pdf

    """
    __doc__ = __doc__.format(params=coord.SphericalCoordinatesBase.\
                                          _init_docstring_param_templ.\
                                          format(lonnm='Lambda', latnm='Beta'))

    def __init__(self, *args, **kwargs):
        super(OrphanCoordinates, self).__init__()

        if len(args) == 1 and len(kwargs) == 0 and \
            isinstance(args[0], coord.SphericalCoordinatesBase):
            newcoord = args[0].transform_to(self.__class__)
            self.Lambda = newcoord.Lambda
            self.Beta = newcoord.Beta
            self._distance = newcoord._distance
        else:
            super(OrphanCoordinates, self)._initialize_latlon('Lambda', 'Beta', \
                False, args, kwargs, anglebounds=((0, 360), (-90,90)))

    def __repr__(self):
        if self.distance is not None:
            diststr = ', Distance={0:.2g} {1!s}'.format(self.distance._value, 
                                                        self.distance._unit)
        else:
            diststr = ''

        msg = "<{0} Lambda={1:.5f} deg, Beta={2:.5f} deg{3}>"
        return msg.format(self.__class__.__name__, self.Lambda.degrees,
                          self.Beta.degrees, diststr)

    @property
    def lonangle(self):
        return self.Lambda

    @property
    def latangle(self):
        return self.Beta

# Define the Euler angles
phi = radians(128.79)
theta = radians(54.39)
psi = radians(90.70)

rot11 = cos(psi)*cos(phi)-cos(theta)*sin(phi)*sin(psi)
rot12 = cos(psi)*sin(phi)+cos(theta)*cos(phi)*sin(psi)
rot13 = sin(psi)*sin(theta)
rot21 = -sin(psi)*cos(phi)-cos(theta)*sin(phi)*cos(psi)
rot22 = -sin(psi)*sin(phi)+cos(theta)*cos(phi)*cos(psi)
rot23 = cos(psi)*sin(theta)
rot31 = sin(theta)*sin(phi)
rot32 = -sin(theta)*cos(phi)
rot33 = cos(theta)

rotation_matrix = np.array([[rot11, rot12, rot13], 
                            [rot21, rot22, rot23],
                            [rot31, rot32, rot33]])

# Galactic to Orphan coordinates
@transformations.transform_function(coord.GalacticCoordinates, OrphanCoordinates)
def galactic_to_orphan(galactic_coord):
    """ Compute the transformation from Galactic spherical to Orphan coordinates. 
    """

    l = galactic_coord.l.radians
    b = galactic_coord.b.radians

    X = cos(b)*cos(l)
    Y = cos(b)*sin(l)
    Z = sin(b)

    # Calculate X,Y,Z,distance in the Orphan system
    Xs, Ys, Zs = rotation_matrix.dot(np.array([X, Y, Z]))
    Zs = -Zs

    # Calculate the angular coordinates lambda,beta
    Lambda = degrees(np.arctan2(Ys,Xs))
    if Lambda<0:
        Lambda += 360

    Beta = degrees(np.arcsin(Zs/np.sqrt(Xs*Xs+Ys*Ys+Zs*Zs)))

    return OrphanCoordinates(Lambda, Beta, distance=galactic_coord.distance, 
                          unit=(u.degree, u.degree))

@transformations.transform_function(OrphanCoordinates, coord.GalacticCoordinates)
def orphan_to_galactic(orphan_coord):
    L = orphan_coord.Lambda.radians
    B = orphan_coord.Beta.radians

    Xs = cos(B)*cos(L)
    Ys = cos(B)*sin(L)
    Zs = sin(B)
    Zs = -Zs

    X, Y, Z = rotation_matrix.T.dot(np.array([Xs, Ys, Zs]))

    l = degrees(np.arctan2(Y,X))
    b = degrees(np.arcsin(Z/np.sqrt(X*X+Y*Y+Z*Z)))

    if l<0:
        l += 360

    return coord.GalacticCoordinates(l, b, distance=orphan_coord.distance, 
                                     unit=(u.degree, u.degree))

def distance_to_orphan_plane(ra, dec, heliocentric_distance):
    """ Given an RA, Dec, and Heliocentric distance, compute the distance
        to the midplane of the Orphan plane 

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
    orphan_coords = eq_coords.transform_to(OrphanCoordinates)
    orphan_coords.distance = coord.Distance(heliocentric_distance, unit=u.kpc)

    Z_orp_sol = orphan_coords.distance.kpc * np.sin(orphan_coords.Beta.radians)

    return Z_orp_sol

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