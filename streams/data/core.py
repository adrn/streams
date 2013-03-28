# coding: utf-8

""" Classes for accessing various data related to this project. """

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import numexpr
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.coordinates as coord
import astropy.units as u
from scipy import interpolate

# Project
from ..util import project_root

__all__ = ["LM10", "LINEAR", "QUEST", "SgrCen", "SgrSnapshot"]

def _make_npy_file(ascii_file, overwrite=False, ascii_kwargs=dict()):
    """ Make a .npy version of the given ascii file data.

        Parameters
        ----------
        ascii_file : str
            The full path to the ascii file to convert.
        overwrite : bool (optional)
            If True, will overwrite any existing npy files and regenerate
            using the latest ascii data.
        ascii_kwargs : dict
            A dictionary of keyword arguments to be passed to ascii.read().
    """

    filename, ext = os.path.splitext(ascii_file)
    npy_filename = filename + ".npy"

    if os.path.exists(npy_filename) and overwrite:
        os.remove(npy_filename)
    elif os.path.exists(npy_filename) and not overwrite:
        return npy_filename

    data = ascii.read(ascii_file, **ascii_kwargs)
    np.save(npy_filename, np.array(data))
    return npy_filename

class StreamData(object):

    def _set_coordinates(self, ra_name, dec_name):
        """ Given the name of the right ascension column and the name of the
            declination column, set the 'ra' and 'dec' attributes as astropy
            coordinates objects.
        """
        icrs = [coord.ICRSCoordinates(ra,dec,unit=(u.degree,u.degree)) \
                for ra,dec in zip(self.data[ra_name], self.data[dec_name])]

        self.ra = [c.ra for c in icrs]
        self.dec = [c.dec for c in icrs]

class LM10(StreamData):

    data = None

    def __init__(self, overwrite=False):
        """ Read in simulation data from Law & Majewski 2010.

            Parameters
            ----------
            overwrite : bool (optional)
                If True, will overwrite any existing npy files and regenerate
                using the latest ascii data.
        """

        if LM10.data == None:
            dat_filename = os.path.join(project_root, "data", "simulation", \
                                    "SgrTriax_DYN.dat")

            npy_filename = _make_npy_file(dat_filename, overwrite=overwrite)
            LM10.data = np.load(npy_filename)

        self._set_coordinates("ra", "dec")

class LINEAR(StreamData):

    data = None

    def __init__(self, overwrite=False):
        """ Read in LINEAR RR Lyrae catalog from Branimir. """

        if LINEAR.data == None:
            txt_filename = os.path.join(project_root, "data", "catalog", \
                                        "LINEAR_RRab.txt")

            npy_filename = _make_npy_file(txt_filename, overwrite=overwrite)
            LINEAR.data = np.load(npy_filename)

        self._set_coordinates("ra", "dec")

class QUEST(StreamData):

    data = None

    def __init__(self, overwrite=False):
        """ Read in QUEST RR Lyrae from Vivas et al. 2004 """

        if QUEST.data == None:
            fits_filename = os.path.join(project_root, "data", "catalog", \
                                        "quest_vivas2004_RRL.fits")

            hdulist = fits.open(fits_filename)
            QUEST.data = hdulist[1].data

        self._set_coordinates("RAJ2000", "DEJ2000")

class SgrData(StreamData):
    
    def __init__(self, overwrite=False):
        """ Read in Sgr satellite center simulation data from Kathryn 
            
            Note that _data is the *unscaled, raw data* from the ASCII
            file!
        
        """

        # Scale Sgr simulation data to physical units
        self._r_scale = 0.63 # kpc
        self._v_scale = (41.27781037*u.km/u.s).to(u.kpc/u.Myr).value # kpc/Myr
        self._t_scale = 14.9238134129 # Myr
        
        self.r_unit = u.kpc
        self.t_unit = u.Myr
        self.v_unit = self.r_unit / self.t_unit
    
    def _name_to_unit(self, name):
        """ Map a column name to a unit object. """
        
        if name in ["t","dt","tub"]:
            return (self._t_scale, self.t_unit)
        elif name in ["x","y","z"]:
            return (self._r_scale, self.r_unit)
        elif name in ["vx","vy","vz"]:
            return (self._v_scale, self.v_unit)
        elif name in ["s1", "s2", "m"]:
            return (1., 1.)
        else:
            raise NameError("Unsure of units for name '{0}'".format(name))
    
    def _set_xyz_vxyz(self):
        self.xyz = np.array([self.x,
                             self.y,
                             self.z])*self.r_unit
                               
        self.vxyz = np.array([self.vx,
                              self.vy,
                              self.vz])*self.v_unit
    
    def _init_data(self):
        """ Using a data array (read in from a .npy binary file), create the 
            aliases .x, .y, etc. for accessing the data.
        """
        
        for name in self._data.dtype.names:
            scale,unit = self._name_to_unit(name)
            setattr(self, name, self._data[name]*scale)
        
        self._set_xyz_vxyz()
    
    def __len__(self):
        return len(self._data)
    
class SgrCen(SgrData):

    _data = None

    def __init__(self, overwrite=False):
        """ Read in Sgr satellite center simulation data from Kathryn """
        super(SgrCen, self).__init__()

        if SgrCen._data == None:
            txt_filename = os.path.join(project_root, 
                                        "data",
                                        "simulation", 
                                        "SGR_CEN")

            npy_filename = _make_npy_file(txt_filename, 
                                          overwrite=overwrite,
                                          ascii_kwargs=dict(names=["t","dt","x","y","z","vx","vy","vz"]))
            SgrCen._data = np.load(npy_filename)
            
        self._init_data()
    
    def interpolate(self, ts):
        """ Interpolate the SgrCen data onto the specified time grid. 
            
            Parameters
            ----------
            ts : astropy.units.Quantity
                The new grid of times to interpolate on to.
        
        """
        
        if not isinstance(ts, u.Quantity):
            raise TypeError("New time grid must be an Astropy Quantity object.")
        
        ts = ts.to(self.t_unit).value
        
        for name in SgrCen._data.dtype.names:
            if name == "t":
                self.t = ts
                continue
            elif name == "dt":
                self.dt = None
            
            t = self._data["t"]*self._t_scale
            
            scale,unit = self._name_to_unit(name)
            d = self._data[name]*scale
            setattr(self, name, 
                    interpolate.interp1d(t, 
                                         d, 
                                         kind='cubic')(ts))
        
        self._set_xyz_vxyz()
    
class SgrSnapshot(SgrData):

    def __init__(self, overwrite=False, num=None, expr=""):
        """ Read in Sgr simulation snapshop for individual particles

            Parameters
            ----------
            num : int
                If None, load all stars, otherwise randomly sample 'num' 
                particles from the snapshot data.
            expr : str (optional)
                A selection expression to be fed in to numexpr. For
                example, to select only unbound particles, use 
                'tub > 0'.
        """
        super(SgrSnapshot, self).__init__()
        
        txt_filename = os.path.join(project_root, "data", "simulation", \
                                    "SGR_SNAP")

        npy_filename = _make_npy_file(txt_filename, \
                                      overwrite=overwrite, \
                                      ascii_kwargs=dict(names=["m","x","y","z","vx","vy","vz","s1", "s2", "tub"]))
        self._data = np.load(npy_filename)
        self._init_data()

        if len(expr.strip()) > 0:
            idx = numexpr.evaluate(str(expr), self.__dict__)
            self._data = self._data[idx]
        
        if num != None and num > 0:
            idx = np.random.randint(0, len(self._data), num)
            self._data = self._data[idx]
        
        # Do this again now that we've selected out the rows we want
        self._init_data()

