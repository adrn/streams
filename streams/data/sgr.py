# coding: utf-8

""" Classes for accessing simulation data related to Sagittarius. """

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy

# Third-party
import numpy as np
import numexpr
import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table, Column
import astropy.coordinates as coord
import astropy.units as u
from scipy import interpolate

# Project
from ..util import project_root
from .gaia import rr_lyrae_add_observational_uncertainties
from ..plot.data import scatter_plot_matrix
from .core import StreamData, _make_npy_file
from ..simulation import TestParticle, TestParticleOrbit

__all__ = ["LM10", "SgrCen", "SgrSnapshot"]

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

class KVJSgrData(Table):
    
    def __init__(self, filename, column_names, overwrite_npy=False, 
                 mask_expr=None, N=None):
        """ Read in Sgr satellite center simulation data from Kathryn. 
            Relevant for either SGR_CEN or SGR_SNAP.
            
            Parameters
            ----------
        """
        
        # If it doesn't already exist, create a .npy binary version of the
        #   ascii data -- loads much faster
        npy_filename = _make_npy_file(filename, 
                                      overwrite=overwrite_npy,
                                      ascii_kwargs=dict(names=column_names))
        data = np.load(npy_filename)
        
        if mask_expr != None and len(mask_expr.strip()) > 0:
            idx = numexpr.evaluate(str(mask_expr), data)
            data = data[idx]
        
        if N != None and N > 0:
            idx = np.random.randint(0, len(data), N)
            data = data[idx]
            
        super(KVJSgrData, self).__init__(data)
                
        self.r_unit = u.kpc
        self.t_unit = u.Myr
        self.v_unit = self.r_unit / self.t_unit
        
        for colname in self.colnames:
            scale,unit = self._name_to_unit(colname)
            self[colname] *= scale
            self[colname].units = unit
        
        self._phase_space_coord_names = ["x","y","z","vx","vy","vz"]
        
    def _name_to_unit(self, name):
        """ Map a column name to a unit object. """
        
        # Scale Sgr simulation data to physical units       
        r_scale = 0.63
        v_scale = (41.27781037*u.km/u.s).to(u.kpc/u.Myr).value
        t_scale = 14.9238134129
        
        if name in ["t","dt","tub"]:
            return (t_scale, self.t_unit)
        elif name in ["x","y","z"]:
            return (r_scale, self.r_unit)
        elif name in ["vx","vy","vz"]:
            return (v_scale, self.v_unit)
        elif name in ["s1", "s2", "m"]:
            return (1, None)
        else:
            raise NameError("Unsure of units for name '{0}'".format(name))
    
    @property
    def xyz(self):
        return np.array([np.array(self["x"]),
                         np.array(self["y"]),
                         np.array(self["z"])])*self.r_unit
    
    @property
    def vxyz(self):
        return np.array([np.array(self["vx"]),
                         np.array(self["vy"]),
                         np.array(self["vz"])])*self.v_unit
    
    def as_particles(self):
        """ Return a list of particles from the position and velocity 
            vectors.
        """
        from ..simulation import TestParticle
        
        r = np.zeros((len(self), 3))
        r[:,0] = np.array(self["x"])
        r[:,1] = np.array(self["y"])
        r[:,2] = np.array(self["z"])
        
        v = np.zeros((len(self), 3))
        v[:,0] = np.array(self["vx"])
        v[:,1] = np.array(self["vy"])
        v[:,2] = np.array(self["vz"])
        
        return TestParticle(r*self.r_unit, v*self.v_unit)

class SgrCen(KVJSgrData):
    
    def __init__(self, overwrite_npy=False):
        
        # Find and read in text file
        txt_filename = os.path.join(project_root, 
                                    "data",
                                    "simulation", 
                                    "SGR_CEN")
        colnames = ["t","dt","x","y","z","vx","vy","vz"]
        
        super(SgrCen, self).__init__(filename=txt_filename, 
                                     column_names=colnames,
                                     overwrite_npy=overwrite_npy)
        
        self.dt = self["dt"][0]
    
    def as_orbit(self):
        """ Return a TestParticleOrbit object. """
        
        r = np.zeros((len(self), 3))
        r[:,0] = np.array(self["x"])
        r[:,1] = np.array(self["y"])
        r[:,2] = np.array(self["z"])
        
        v = np.zeros((len(self), 3))
        v[:,0] = np.array(self["vx"])
        v[:,1] = np.array(self["vy"])
        v[:,2] = np.array(self["vz"])
        
        t = np.array(self["t"])
        return TestParticleOrbit(t*self.t_unit, r*self.r_unit, v*self.v_unit)
        
class SgrSnapshot(KVJSgrData):

    def __init__(self, N=None, expr="", overwrite_npy=False):
        """ Read in Sgr simulation snapshop for individual particles

            Parameters
            ----------
            N : int
                If None, load all stars, otherwise randomly sample 'N' 
                particles from the snapshot data.
            expr : str (optional)
                A selection expression to be fed in to numexpr. For
                example, to select only unbound particles, use 
                'tub > 0'.
        """
        
        # Find and read in text file
        txt_filename = os.path.join(project_root, "data", 
                                    "simulation", "SGR_SNAP")
        colnames = ["m","x","y","z","vx","vy","vz","s1", "s2", "tub"]
        
        super(SgrSnapshot, self).__init__(filename=txt_filename, 
                                          column_names=colnames,
                                          overwrite_npy=overwrite_npy,
                                          mask_expr=expr,
                                          N=N)
    
    def add_errors(self):
        """ Add observational errors to the data from GAIA error estimates
            
            Returns a *new* SgrSnapshot object.
        """
        
        coords_w_errors = rr_lyrae_add_observational_uncertainties(
                            self["x"].data*self.r_unit,
                            self["y"].data*self.r_unit,
                            self["z"].data*self.r_unit,
                            self["vx"].data*self.v_unit,
                            self["vy"].data*self.v_unit,
                            self["vz"].data*self.v_unit
                         )
        
        columns = []
        for colname in self.colnames:
            if colname in self._phase_space_coord_names:
                ii = self._phase_space_coord_names.index(colname)
                columns.append(Column(data=coords_w_errors[ii], 
                                      units=coords_w_errors[ii].unit,
                                      name=colname))
            else:
                columns.append(self[colname])
        
        # MAJOR hack here....
        new_table = copy.copy(self)
        new_table2 = Table(columns, names=self.colnames)
        new_table._init_from_table(new_table2, self.colnames, 
                                   [self.dtype[ii] for ii in range(len(self.colnames))], 
                                   len(self.colnames), 
                                   True)
        return new_table
    
    