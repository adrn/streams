# coding: utf-8

""" Tools for reading in simulation config files. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u

def read(file):
    """ Read in configuration parameters for the simulation from the 
        given text file. Each line should be formatted as follows:
            (data type) variable_name: value
        
        where (data type) should be:
            (I) if integer
            (F) if float
            (U) if a number with units
            (B) if boolean
            (S) if string
        
        Parameters
        ----------
        file : str or file-like
            Path to configuration file or file-like object.
    """
    
    try:
        # file is a file-like object
        file_lines = file.read().split("\n")
    except AttributeError:
        # file isn't a file object, try openining it as if it were a filename
        with open(filename, "r") as f:
            file_lines = f.readlines()
    
    config = dict()
    for ii,line in enumerate(file_lines):
        # comment or blank line
        if line.startswith("#") or len(line.strip()) == 0:
            continue
        
        key,val = map(lambda x: x.strip(), line.split(":"))
        
        if "#" in val:
            val,comment = val.split("#")
            val = val.strip()
        
        if key.startswith("(I)"):
            # integer
            config[key.split()[1]] = int(val)
        elif key.startswith("(F)"):
            # float
            config[key.split()[1]] = float(val)
        elif key.startswith("(U)"):
            # number with units
            num,unit = val.split()
            config[key.split()[1]] = float(num)*u.Unit(unit)
        elif key.startswith("(S)"):
            # string
            config[key.split()[1]] = str(val)
        elif key.startswith("(B)"):
            # boolean
            config[key.split()[1]] = val.lower() in ("yes", "true", "t", 
                                                     "y", "1", "duh")
        else:
            raise ValueError("Unknown datatype for line {0}: '{1}'"
                             .format(ii+1,line))
    
    return config

def test_read():
    import cStringIO as StringIO
    
    file = """(I) particles : 100 # number of particles
              (U) dt : 1. Myr # timestep for back integration
              (B) with_errors : yes
              (S) description : blah blah blah"""
    
    f = StringIO.StringIO(file)
    config = read(f)
    
    assert config["particles"] == 100
    assert config["dt"] == (1.*u.Myr)
    assert config["with_errors"] == True