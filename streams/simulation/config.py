# coding: utf-8

""" Tools for reading in simulation config files. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
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
        config[k] = v
    
    return config

def write(filename, clobber=False):
    """ Write all defaults into a configuration file.
    
        Parameters
        ----------
        filename : str 
            File to write the default values to.
        clobber : bool (optional)
            If the file exists, overwrite it.
    """
    
    config_file = []
    config_file.append("(I) particles: 100")
    config_file.append("(U) dt: 5. Myr # timestep")
    config_file.append("(B) observational_errors: yes # add observational errors")
    config_file.append("(S) expr: tub > 0.")
    config_file.append("(L,S) model_parameters: q1 qz v_halo phi")
    config_file.append("(I) seed: 42")
    config_file.append("(I) walkers: 128")
    config_file.append("(I) burn_in: 50")
    config_file.append("(I) steps: 100")
    config_file.append("(B) mpi: yes")
    config_file.append("(B) make_plots: no")
    config_file.append("(S) output_path: /tmp/")
    
    if os.path.exists(filename) and clobber:
        os.remove(filename)
    elif os.path.exists(filename) and not clobber:
        raise IOError("File exists and you don't want to clobber it!")
        
    f = open(filename, "w")
    f.write("\n".join(config_file))
    f.close()
    