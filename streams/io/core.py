# coding: utf-8

""" Code for helping to select stars from the nearby Sgr wraps. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import gc

# Third-party
import h5py
import numpy as np
import numexpr
import astropy.units as u
from astropy.io import ascii
from astropy.table import vstack, Table, Column
import astropy.coordinates as coord

# Project
from ..coordinates.frame import heliocentric, galactocentric
from ..dynamics import Particle, ObservedParticle, Orbit

__all__ = ["read_table", "read_hdf5"]

def read_table(filename, expr=None, N=None):
    _table = np.genfromtxt(filename, names=True)

    if expr is not None:
        idx = numexpr.evaluate(str(expr), _table)
        _table = _table[idx]

    if N is not None and N > 0:
        np.random.shuffle(_table)
        _table = _table[:min(N,len(_table))]

    return _table

def read_hdf5(h5file, nparticles=None):
    """ Read particles and satellite from a given HDF5 file. """

    return_dict = dict()
    with h5py.File(h5file, "r") as f:
        try:
            ptcl = f["particles"]
            satl = f["satellite"]
        except KeyError:
            raise ValueError("Invalid HDF5 file. Missing 'particles' or "
                             "'satellite' group.")

        if nparticles is None:
            nparticles = len(ptcl["data"].value)

        true_tub = ptcl["tub"].value
        if "error" in ptcl.keys():
            p = ObservedParticle(ptcl["data"].value[:nparticles].T,
                                 ptcl["error"].value[:nparticles].T,
                                 frame=heliocentric,
                                 units=[u.Unit(x) for x in ptcl["units"]])

            true_p = Particle(ptcl["true_data"].value[:nparticles].T,
                              frame=heliocentric,
                              units=[u.Unit(x) for x in ptcl["units"]])
            true_p.tub = true_tub[:nparticles]
            return_dict["true_particles"] = true_p
        else:
            p = Particle(ptcl["data"].value.T,
                         frame=heliocentric,
                         units=[u.Unit(x) for x in ptcl["units"]])
            p.tub = true_tub
        return_dict["particles"] = p

        if "error" in satl.keys():
            s = ObservedParticle(satl["data"].value.T, satl["error"].value.T,
                                 frame=heliocentric,
                                 units=[u.Unit(x) for x in satl["units"]])
            return_dict["true_satellite"] = Particle(satl["true_data"].value.T,
                                                     frame=heliocentric,
                                                     units=[u.Unit(x) for x in satl["units"]])

        else:
            s = Particle(satl["data"].value.T,
                         frame=heliocentric,
                         units=[u.Unit(x) for x in satl["units"]])
        s.m = satl["m"].value
        s.v_disp = satl["v_disp"].value
        return_dict["satellite"] = s

        if "simulation" in f.keys():
            return_dict["t1"] = float(f["simulation"]["t1"].value)
            return_dict["t2"] = float(f["simulation"]["t2"].value)

    return return_dict