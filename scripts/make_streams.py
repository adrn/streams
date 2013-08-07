#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" Script for forward-integrating satellite disruption to create 
    stellar streams.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle

# Project


# Create logger
logger = logging.getLogger(__name__)

def spherical(x):
    try:
        N = len(x)
    except TypeError: # float
        N = 1
    
    xyz = np.zeros((N,3))
    
    phi = np.random.uniform(0., 2*np.pi, size=N)
    theta = np.arccos(np.random.uniform(-1., 1., size=N))
    
    # x, y, and z components of the vector of magnitude x
    xyz[:,0] = x * np.sin(theta) * np.cos(phi)
    xyz[:,1] = x * np.sin(theta) * np.sin(phi)
    xyz[:,2] = x * np.cos(theta)
    
    return xyz

def plummer_particles(N=100, M=2.5E8*u.M_sun, a=1*u.kpc):
    """ Sample N stars from an elliptical galaxy of the specified
        mass, assuming a Plummer model 
        
            $\Phi = - G M / \sqrt{r^2 + a^2}
        
        Note:
            For now, assumes all particle masses = 1 M_sun, but could
            in principle sample from some mass distribution.
        
    """
    
    # fractional enclosed mass
    m_M = np.random.uniform(size=N)
    
    # sample positions
    radius = a / np.sqrt(m_M**(-2./3) - 1)
    xyz = spherical(radius)
    
    # sample velocities, based on positions
    v_mags = np.zeros(N)
    for ii in range(N):
        xx = 0.0
        yy = 0.1
        
        while yy > xx*xx*(1.0-xx*xx)**(3.5):
            xx = np.random.uniform()
            yy = np.random.uniform()*0.1
        
        v_mags[ii] = (xx * np.sqrt(2.0*G*M) * (radius[ii]**2 + a**2) ** -0.25).to(u.km/u.s).value
    
    v_xyz = spherical(v_mags)
    x_v = np.hstack((xyz,v_xyz))
    
    return x_v

def main():
    np.random.seed(42)
    
    plummer_particles
            
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", 
                    default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
                    default=False, help="Be quiet! (default = False)")
    parser.add_argument("-f", "--file", dest="file", default="streams.cfg", 
                    help="Path to the configuration file to run with.")
    parser.add_argument("-n", "--name", dest="job_name", default=None, 
                    help="Name of the output.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        main(args.file, args.job_name)
        
    except:
        raise
        sys.exit(1)
    
    sys.exit(0)
