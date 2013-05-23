# coding: utf-8

""" Classes for accessing simulation data related to Sagittarius. """

from __future__ import division, print_function

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
from .core import _make_npy_file
from ..observation.gaia import rr_lyrae_add_observational_uncertainties
from ..util import project_root
from ..plot.data import scatter_plot_matrix
from ..nbody import Particle, ParticleCollection, Orbit, OrbitCollection

__all__ = ["LM10Cen", "LM10Snapshot", "SgrCen", "SgrSnapshot", "read_lm10"]

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
        self.angle_unit = u.degree
        
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
        
        r = np.zeros((len(self), 3))
        r[:,0] = np.array(self["x"])
        r[:,1] = np.array(self["y"])
        r[:,2] = np.array(self["z"])
        
        v = np.zeros((len(self), 3))
        v[:,0] = np.array(self["vx"])
        v[:,1] = np.array(self["vy"])
        v[:,2] = np.array(self["vz"])
        
        return ParticleCollection(r=r*self.r_unit, 
                                  v=v*self.v_unit,
                                  m=np.ones(len(r))*u.M_sun,
                                  units=[self.t_unit,self.r_unit,self.v_unit, u.M_sun])

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
        """ Return an OrbitCollection object. """
        
        r = np.zeros((len(self), 3))
        r[:,0] = np.array(self["x"])
        r[:,1] = np.array(self["y"])
        r[:,2] = np.array(self["z"])
        
        v = np.zeros((len(self), 3))
        v[:,0] = np.array(self["vx"])
        v[:,1] = np.array(self["vy"])
        v[:,2] = np.array(self["vz"])
        
        t = np.array(self["t"])
        return Orbit(t=t*self.t_unit,
                     r=r*self.r_unit,
                     v=v*self.v_unit,
                     m=np.ones(len(r))*u.M_sun)
        
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
    

class LM10Snapshot(KVJSgrData):

    def __init__(self, N=None, expr="", overwrite_npy=False):
        """ Read in Sgr simulation snapshop from David Law's 2010 simulation

            Parameters
            ----------
            N : int
                If None, load all stars, otherwise randomly sample 'N' 
                particles from the snapshot data.
            expr : str (optional)
                A selection expression to be fed in to numexpr. For
                example, to select only unbound particles, use 
                'Pcol > -1'.
        """
        
        # Find and read in text file
        txt_filename = os.path.join(project_root, "data", "simulation", \
                                    "SgrTriax_DYN.dat")
        colnames = "lambda beta ra dec l b xgc ygc zgc xsun ysun zsun x4 y4 z4 u v w dist vgsr mul mub mua mud Pcol Lmflag".split()
        
        super(LM10Snapshot, self).__init__(filename=txt_filename, 
                                          column_names=colnames,
                                          overwrite_npy=overwrite_npy,
                                          mask_expr=expr,
                                          N=N)
        
        self.rename_column('xgc', "x")
        self.rename_column('ygc', "y")
        self.rename_column('zgc', "z")
        self["x"] = -self["x"]
        
        self.rename_column('u', "vx")
        self.rename_column('v', "vy")
        self.rename_column('w', "vz")
        self["vx"] = -self["vx"]
        
    def _name_to_unit(self, name):
        """ Map a column name to a unit object. """
        
        if name in ["lambda", "beta", "ra", "dec", "l", "b"]:
            return (1., self.angle_unit)
        elif name in ["xgc","ygc","zgc","xsun","ysun","zsun", "dist", "x", "y", "z"]:
            return (1., self.r_unit)
        elif name in "x4 y4 z4 Pcol Lmflag".split():
            return (1.,1.)
        elif name in ["vgsr", "u", "v", "w", "vx", "vy", "vz"]:
            return (1., u.km/u.s)
        elif name in ["mul", "mub", "mua", "mud"]:
            return (1., u.mas/u.yr)
        else:
            raise NameError("Unsure of units for name '{0}'".format(name))

class LM10Cen(KVJSgrData):
    
    def __init__(self, overwrite_npy=False):
        
        # Find and read in text file
        txt_filename = os.path.join(project_root, 
                                    "data",
                                    "simulation", 
                                    "SgrTriax_orbit.dat")
        colnames = ["t","lambda_sun","beta_sun","ra","dec","x_sun","y_sun",\
                    "z_sun","x_gc","y_gc","z_gc","dist","vgsr"]
        
        super(LM10Cen, self).__init__(filename=txt_filename, 
                                     column_names=colnames,
                                     overwrite_npy=overwrite_npy)
        
        #self["t"] += np.fabs(min(self["t"]))
        self.dt = self["t"][1]-self["t"][0]
    
    def _name_to_unit(self, name):
        """ Map a column name to a unit object. """
        
        if name in ["lambda_sun", "beta_sun", "ra", "dec", "l", "b"]:
            return (1., self.angle_unit)
        elif name in ["x_gc","y_gc","z_gc","x_sun","y_sun","z_sun", "dist", "x", "y", "z"]:
            return (1., self.r_unit)
        elif name in "x4 y4 z4 Pcol Lmflag".split():
            return (1.,1.)
        elif name in ["vgsr", "u", "v", "w", "vx", "vy", "vz"]:
            return (1., u.km/u.s)
        elif name in ["mul", "mub", "mua", "mud"]:
            return (1., u.mas/u.yr)
        elif name in ["t"]:
            return (1000., u.Myr)
        else:
            raise NameError("Unsure of units for name '{0}'".format(name))
    
    def today(self):
        return self.as_orbit()[-1]
    
    def as_orbit(self):
        """ Return an Orbit object. """
        
        r = np.zeros((len(self), 3))
        r[:,0] = -np.array(self["x_gc"])
        r[:,1] = np.array(self["y_gc"])
        r[:,2] = np.array(self["z_gc"])
        
        v = np.zeros((len(self), 3))
        v[:,0] += -(-230.*u.km/u.s).to(u.kpc/u.Myr).value
        v[:,1] += (-35.*u.km/u.s).to(u.kpc/u.Myr).value
        v[:,2] += (195.*u.km/u.s).to(u.kpc/u.Myr).value
        
        t = np.array(self["t"])
        return OrbitCollection(t=t*self.t_unit,
                               r=r*self.r_unit,
                               v=v*self.v_unit,
                               m=np.ones(len(r))*u.M_sun,
                               units=[self.t_unit,self.r_unit,self.v_unit, u.M_sun])
        
def read_lm10(N=None, expr=None, dt=1.):
    """ """
    
    # Read in the file describing the orbit of the satellite.
    sat_filename = os.path.join(project_root, 
                                "data",
                                "simulation", 
                                "SgrTriax_orbit.dat")
    sat_colnames = ["t","lambda_sun","beta_sun","ra","dec", \
                    "x_sun","y_sun","z_sun","x_gc","y_gc","z_gc", \
                    "dist","vgsr"]
    
    satellite_data = ascii.read(sat_filename, names=sat_colnames)
    satellite_data = satellite_data[satellite_data["t"] <= 0.]
    satellite_data.rename_column("lambda_sun", "lambda")
    satellite_data.rename_column("beta_sun", "beta")
    
    satellite_data.rename_column("x_gc", "x")
    satellite_data["x"] = -satellite_data["x"]
    satellite_data.rename_column("y_gc", "y")
    satellite_data.rename_column("z_gc", "z")
    
    # initial conditions, or, position of the satellite today
    r = [[satellite_data[-1]['x'], satellite_data[-1]['y'], satellite_data[-1]['z']]]*u.kpc # kpc
    v = [[230., -35., 195.]]*u.km/u.s
    
    satellite = ParticleCollection(r=r, v=v, m=[2.5E8]*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    t1 = (satellite_data[-1]['t'] + abs(satellite_data[0]['t'])) * 1000. # Myr
    t2 = 0.
    time_grid = np.arange(t1, t2, -dt)*u.Myr
    
    # Read in particle data -- a snapshot of particle positions, velocities at
    #   the end of the simulation
    particle_filename = os.path.join(project_root, 
                                     "data",
                                     "simulation", 
                                     "SgrTriax_DYN.dat")
    particle_colnames = ["Lambda", "Beta", "ra", "dec", "l", "b", \
                         "xgc", "ygc", "zgc", "xsun", "ysun", "zsun", \
                         "x4", "y4", "z4", "u", "v", "w", "dist", "vgsr", \
                         "mul", "mub", "mua", "mud", "Pcol", "Lmflag"]
    
    particle_data = ascii.read(particle_filename, names=particle_colnames)
    particle_data.add_column(Column(data=-np.array(particle_data["xgc"]), name="x"))
    particle_data.add_column(Column(data=np.array(particle_data["ygc"]), name="y"))
    particle_data.add_column(Column(data=np.array(particle_data["zgc"]), name="z"))
    particle_data.add_column(Column(data=-np.array(particle_data["u"]), name="vx"))
    particle_data.add_column(Column(data=np.array(particle_data["v"]), name="vy"))
    particle_data.add_column(Column(data=np.array(particle_data["w"]), name="vz"))

    if expr != None and len(expr.strip()) > 0:
        idx = numexpr.evaluate(str(expr), particle_data)
        particle_data = particle_data[idx]
    
    if N != None and N > 0:
        idx = np.random.randint(0, len(particle_data), N)
        particle_data = particle_data[idx]
    
    r = np.zeros((len(particle_data), 3))
    r[:,0] = np.array(particle_data["x"])
    r[:,1] = np.array(particle_data["y"])
    r[:,2] = np.array(particle_data["z"])
    
    v = np.zeros((len(particle_data), 3))
    v[:,0] = np.array(particle_data["vx"])
    v[:,1] = np.array(particle_data["vy"])
    v[:,2] = np.array(particle_data["vz"])
        
    particles = ParticleCollection(r=r*u.kpc,
                                   v=v*u.km/u.s,
                                   m=np.zeros(len(r))*u.M_sun,
                                   units=[u.kpc,u.Myr,u.M_sun])
    
    return time_grid, satellite, particles