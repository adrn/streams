# coding: utf-8

""" Classes for accessing simulation data for Sgr-like streams with
    different mass progenitors.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import numexpr
import astropy.io.ascii as ascii
from astropy.table import Column
import astropy.units as u
from astropy.constants import G

# Project
from .. import usys
from ..dynamics import Particle, Orbit
from .core import read_table
from ..util import project_root
from ..coordinates.frame import galactocentric
from ..integrate import LeapfrogIntegrator
from ..potential.lm10 import LawMajewski2010

__all__ = ["SgrSimulation", "SgrSimulationDH"]

def _units_from_file(scfpar):
    """ Generate a unit system from an SCFPAR file. """

    with open(scfpar) as f:
        lines = f.readlines()
        length = float(lines[16].split()[0])
        mass = float(lines[17].split()[0])

    GG = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
    X = (GG / length**3 * mass)**-0.5

    length_unit = u.Unit("{0} kpc".format(length))
    mass_unit = u.Unit("{0} M_sun".format(mass))
    time_unit = u.Unit("{:08f} Myr".format(X))

    return dict(length=length_unit,
                mass=mass_unit,
                time=time_unit)

class SgrSimulation(object):

    def __init__(self, mass):
        self._path = os.path.join(project_root, "data", "simulation", "Sgr")

        self.particle_filename = os.path.join(self._path,mass,"SNAP")
        self.particle_columns = ("x","y","z","vx","vy","vz")

        self._units = _units_from_file(os.path.join(self._path,mass,"SCFPAR"))
        self.particle_units = [self._units["length"]]*3 + \
                              [self._units["length"]/self._units["time"]]*3

        self.true_potential = LawMajewski2010()

        self.mass = mass
        self.t1 = (4.189546E+02 * self._units["time"]).decompose(usys).value
        self.t2 = 0

    def raw_particle_table(self, N=None, expr=None):
        tbl = read_table(self.particle_filename, N=N, expr=expr)
        return tbl

    def particles(self, N=None, expr=None, meta_cols=[], tail_bit=False):
        """ Return a Particle object with N particles selected from the
            simulation with expression expr.

            Parameters
            ----------
            N : int or None (optional)
                Number of particles to return. None or 0 means 'all'
            expr : str (optional)
                Use numexpr to select out only rows that match criteria.
            meta_cols : iterable (optional)
                List of columns to add to meta data.
            tail_bit : bool (optional)
                Compute tail bit or not.
        """
        tbl = self.raw_particle_table(N=N, expr=expr)

        q = []
        for colname,unit in zip(self.particle_columns, self.particle_units):
            q.append(np.array(tbl[colname])*unit)

        meta = dict(expr=expr)
        meta["tub"] = (np.array(tbl["tub"])*self._units["time"]).decompose(usys).value
        for col in meta_cols:
            meta[col] = np.array(tbl[col])

        p = Particle(q, frame=galactocentric, meta=meta)
        p = p.decompose(usys)

        ################################################################
        # HACK: Figure out if in leading or trailing tail
        # separate out the orbit of the satellite from the orbit of the stars
        if tail_bit:
            s = self.satellite()
            X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
            V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
            integrator = LeapfrogIntegrator(self.true_potential._acceleration_at,
                                            np.array(X), np.array(V),
                                            args=(X.shape[0], np.zeros_like(X)))
            ts, rs, vs = integrator.run(t1=self.t1, t2=self.t2, dt=-1.)
            s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
            p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T

            # m_t = -s.mdot*ts + s.m0
            # s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
            # s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
            # r_tide = self.true_potential._tidal_radius(m_t, s_orbit[...,:3])
            # v_disp = s_V * r_tide / s_R

            t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])
            s_orbit = np.array([s_orbit[jj,0] for jj in t_idx])
            p_orbits = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
            # r_tide = np.array([r_tide[jj,0] for jj in t_idx])
            # v_disp = np.array([v_disp[jj,0] for jj in t_idx])

            # instantaneous cartesian basis to project into
            x_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
            y_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
            z_hat = np.cross(x_hat, y_hat)

            # translate to satellite position
            rel_orbits = p_orbits - s_orbit
            rel_pos = rel_orbits[...,:3]
            rel_vel = rel_orbits[...,3:]

            # project onto X
            X = np.sum(rel_pos * x_hat, axis=-1)
            Y = np.sum(rel_pos * y_hat, axis=-1)
            Z = np.sum(rel_pos * z_hat, axis=-1)

            VX = np.sum(rel_vel * x_hat, axis=-1)
            VY = np.sum(rel_vel * y_hat, axis=-1)
            VZ = np.sum(rel_vel * z_hat, axis=-1)

            Phi = np.arctan2(Y, X)
            tail_bit = np.ones(p.nparticles)
            tail_bit[:] = np.nan
            tail_bit[np.cos(Phi) < -0.5] = -1.
            tail_bit[np.cos(Phi) > 0.5] = 1.

            p.meta["tail_bit"] = p.tail_bit = tail_bit
        ################################################################
        else:
            tail_bit = np.ones(p.nparticles)*np.nan

        return p

    def satellite(self):
        """ Return a Particle object with the present-day position of the
            satellite, computed from the still-bound particles.
        """
        expr = "tub==0"
        tbl = self.raw_particle_table(expr=expr)

        q = []
        for colname in self.particle_columns:
            q.append(tbl[colname].tolist())

        q = np.array(q)

        meta = dict(expr=expr)
        v_disp = np.sqrt(np.sum(np.var(q[3:],axis=1)))
        meta["v_disp"] = (v_disp*self.particle_units[-1]).decompose(usys).value
        meta["m"] = float(self.mass)

        m0 = float(self.mass)
        alpha = 3.3*10**(np.floor(np.log10(m0))-4)
        meta['m0'] = m0
        meta['mdot'] = alpha

        q = np.median(q, axis=1)
        p = Particle(q, frame=galactocentric,
                     units=self.particle_units,
                     meta=meta)
        return p.decompose(usys)

    @property
    def satellite_orbit(self):
        """ TODO: """

        orbit_table = np.genfromtxt(os.path.join(self._path,self.mass,"SCFCEN"), names=True)

        q = []
        for colname,unit in zip(self.particle_columns, self.particle_units):
            q.append(np.array(orbit_table[colname])[:,np.newaxis]*unit)

        return Orbit(orbit_table["t"]*self._units["time"], q,
                     frame=galactocentric)

class SgrSimulationDH(object):

    def __init__(self, mass, L="1.0"):
        self._path = os.path.join(project_root, "data", "simulation", "Sgr_DH")

        self._names = "m x y z vx vy vz s1 s2 tub".split()

        m = "M{}e+0{}".format(*mass.split('e'))
        l = "L{}".format(L)

        R = os.listdir(os.path.join(self._path,m))
        if R[0].startswith('.'):
            R = R[1]
        else:
            R = R[0]
        self.root = os.path.join(self._path,m,R,"4.0Gyr",l)
        self.particle_filename = os.path.join(self.root,"SNAP049")
        self.particle_columns = ("x","y","z","vx","vy","vz")

        self._units = _units_from_file(os.path.join(self.root,"SCFPAR"))
        self.particle_units = [self._units["length"]]*3 + \
                              [self._units["length"]/self._units["time"]]*3

        self.true_potential = LawMajewski2010()

        self.mass = mass
        self.t1 = 4000. # Myr
        self.t2 = 0.

    def raw_particle_table(self, N=None, expr=None):
        tbl = read_table(self.particle_filename, N=N, expr=expr, names=self._names, skip_header=1)
        return tbl

    def particles(self, N=None, expr=None, meta_cols=[], tail_bit=False):
        """ Return a Particle object with N particles selected from the
            simulation with expression expr.

            Parameters
            ----------
            N : int or None (optional)
                Number of particles to return. None or 0 means 'all'
            expr : str (optional)
                Use numexpr to select out only rows that match criteria.
            meta_cols : iterable (optional)
                List of columns to add to meta data.
            tail_bit : bool (optional)
                Compute tail bit or not.
        """
        tbl = self.raw_particle_table(N=N, expr=expr)

        q = []
        for colname,unit in zip(self.particle_columns, self.particle_units):
            q.append(np.array(tbl[colname])*unit)

        meta = dict(expr=expr)
        meta["tub"] = (np.array(tbl["tub"])*self._units["time"]).decompose(usys).value
        for col in meta_cols:
            meta[col] = np.array(tbl[col])

        p = Particle(q, frame=galactocentric, meta=meta)
        p = p.decompose(usys)

        ################################################################
        # HACK: Figure out if in leading or trailing tail
        # separate out the orbit of the satellite from the orbit of the stars
        if tail_bit:
            s = self.satellite()
            X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
            V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
            integrator = LeapfrogIntegrator(self.true_potential._acceleration_at,
                                            np.array(X), np.array(V),
                                            args=(X.shape[0], np.zeros_like(X)))
            ts, rs, vs = integrator.run(t1=self.t1, t2=self.t2, dt=-1.)
            s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
            p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T

            # m_t = -s.mdot*ts + s.m0
            # s_R = np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))
            # s_V = np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))
            # r_tide = self.true_potential._tidal_radius(m_t, s_orbit[...,:3])
            # v_disp = s_V * r_tide / s_R

            t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])
            s_orbit = np.array([s_orbit[jj,0] for jj in t_idx])
            p_orbits = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
            # r_tide = np.array([r_tide[jj,0] for jj in t_idx])
            # v_disp = np.array([v_disp[jj,0] for jj in t_idx])

            # instantaneous cartesian basis to project into
            x_hat = s_orbit[...,:3] / np.sqrt(np.sum(s_orbit[...,:3]**2, axis=-1))[...,np.newaxis]
            y_hat = s_orbit[...,3:] / np.sqrt(np.sum(s_orbit[...,3:]**2, axis=-1))[...,np.newaxis]
            z_hat = np.cross(x_hat, y_hat)

            # translate to satellite position
            rel_orbits = p_orbits - s_orbit
            rel_pos = rel_orbits[...,:3]
            rel_vel = rel_orbits[...,3:]

            # project onto X
            X = np.sum(rel_pos * x_hat, axis=-1)
            Y = np.sum(rel_pos * y_hat, axis=-1)
            Z = np.sum(rel_pos * z_hat, axis=-1)

            VX = np.sum(rel_vel * x_hat, axis=-1)
            VY = np.sum(rel_vel * y_hat, axis=-1)
            VZ = np.sum(rel_vel * z_hat, axis=-1)

            Phi = np.arctan2(Y, X)
            tail_bit = np.ones(p.nparticles)
            tail_bit[:] = np.nan
            tail_bit[np.cos(Phi) < -0.5] = -1.
            tail_bit[np.cos(Phi) > 0.5] = 1.

            p.meta["tail_bit"] = p.tail_bit = tail_bit
        ################################################################
        else:
            tail_bit = np.ones(p.nparticles)*np.nan

        return p

    def satellite(self):
        """ Return a Particle object with the present-day position of the
            satellite, computed from the still-bound particles.
        """
        expr = "tub==0"
        tbl = self.raw_particle_table(expr=expr)

        q = []
        for colname in self.particle_columns:
            q.append(tbl[colname].tolist())

        q = np.array(q)

        meta = dict(expr=expr)
        v_disp = np.sqrt(np.sum(np.var(q[3:],axis=1)))
        meta["v_disp"] = (v_disp*self.particle_units[-1]).decompose(usys).value
        meta["m"] = float(self.mass)

        m0 = float(self.mass)
        alpha = 3.3*10**(np.floor(np.log10(m0))-4)
        meta['m0'] = m0
        meta['mdot'] = alpha

        q = np.median(q, axis=1)
        p = Particle(q, frame=galactocentric,
                     units=self.particle_units,
                     meta=meta)
        return p.decompose(usys)
