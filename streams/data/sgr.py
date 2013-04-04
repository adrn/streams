# coding: utf-8

""" Classes for accessing simulation data related to Sagittarius. """

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

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
        
        columns = []
        for name in self.colnames:
            if name == "t":
                columns.append(Column(data=ts, units=self.t_unit, name=name))
                continue
            elif name in ["x","y","z","vx","vy","vz"]:
                data = interpolate.interp1d(self["t"].data, self[name], 
                                            kind='cubic')(ts)
                columns.append(Column(data=data, units=self[name].units, name=name))
            else:
                pass

        return Table(columns)
    
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
        """ """
        
        xyz,vxyz = rr_lyrae_add_observational_uncertainties(self.xyz,self.vxyz)
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]
        
        self.vx = vxyz[0]
        self.vy = vxyz[1]
        self.vz = vxyz[2]
        
        self._set_xyz_vxyz()
    
    def plot_positions(self, **kwargs):
        """ Make a scatter-plot of 3 projections of the positions of the 
            particles in Galactocentric XYZ coordinates.
        """   
        
        labels = [r"${0}_{{GC}}$ [{1}]".format(nm, self.r_unit)
                    for nm in ["X", "Y", "Z"]]
        
        fig,axes = scatter_plot_matrix(self.xyz, labels=labels, **kwargs)
        return fig, axes
        
    def plot_velocities(self, **kwargs):
        """ Make a scatter-plot of 3 projections of the velocities of the 
            particles. 
        """
        
        labels = [r"${0}_{{GC}}$ [{1}]".format(nm, self.r_unit)
                    for nm in ["V^x", "V^y", "V^z"]]
        
        fig,axes = scatter_plot_matrix(self.vxyz, labels=labels, **kwargs)
        return fig, axes