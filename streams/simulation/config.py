# coding: utf-8

""" Tools for reading in simulation config files. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import astropy.units as u

def _parse_key_val(key, val):
    """ Given a key,value pair from a simulation configuration file, parse 
        the value into a typed Python object.
        
        Parameters
        ----------
        key : str
        val : str
    """
    _key = key.lower()
    
    if _key.startswith("(i)"):
        # integer
        k = _key.split()[1]
        v = int(val)
    elif _key.startswith("(f)"):
        # float
        k = _key.split()[1]
        v = float(val)
    elif _key.startswith("(u)"):
        # number with units
        num,unit = val.split()
        k = _key.split()[1]
        v = float(num)*u.Unit(unit)
    elif _key.startswith("(s)"):
        # string
        k = _key.split()[1]
        v = str(val)
    elif _key.startswith("(b)"):
        # boolean
        k = _key.split()[1]
        v = val.lower() in ("yes", "true", "t", "y", "1", "duh")
    elif _key.startswith("(l"):
        # list
        l,element_type = _key.split(",")
        v = val.split()
        k = element_type.split()[1]
    elif _key.startswith("(m"):
        # multi-line
        l,element_type = _key.split(",")
        k,v = _parse_key_val("({0}".format(element_type), 
                             val)
        v = [v]
    else:
        raise ValueError("Unknown datatype in key '{0}'.".format(key))
    
    return k,v

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
            (L,[I,F,U,B,S]) if list, where second letter specifies element type
            (M,[I,F,U,B,S]) if multi-line, where second letter specifies element type
        
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
        with open(file, "r") as f:
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
    
        k,v = _parse_key_val(key,val)
        if config.has_key(k) and isinstance(config[k], list):
            config[k] += v
        elif not config.has_key(k):
            config[k] = v
        else:
            raise ValueError("Unknown derp.")
    
    return config