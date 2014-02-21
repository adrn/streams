# coding: utf-8

""" Code for helping to select stars from the nearby Sgr wraps. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import gc
from random import sample

# Third-party
import h5py
import numpy as np
import numexpr
import astropy.units as u
from astropy.io import ascii
from astropy.table import vstack, Table, Column
import astropy.coordinates as coord
import yaml

# Project
from ..coordinates.frame import heliocentric, galactocentric
from ..dynamics import Particle, ObservedParticle, Orbit
from ..util import OrderedDictYAMLLoader

__all__ = ["read_table", "read_hdf5", "read_config"]

def _check_config_key(config, key, more=""):
    if not config.has_key(key):
        raise KeyError("You must specify the parameter '{}' {}".format(key, more))

def read_config(filename, default_filename=''):
    """ Read in a YAML config file and fille unspecified keys with defaults from
        the specified default_file.

        Parameters
        ----------
        filename : str
        default_filename : str
    """

    # read and load YAML file
    try:
        with open(filename) as f:
            config = yaml.load(f.read(), OrderedDictYAMLLoader)
    except:
        config = yaml.load(filename, OrderedDictYAMLLoader)

    # first make sure the path to the streams project is specified either
    #   as an env var or in the yaml
    if not config.has_key('streams_path'):
        if os.environ.has_key('STREAMSPATH'):
            config['streams_path'] = os.environ["STREAMSPATH"]
        else:
            raise KeyError("Must specify the path to the streams project as 'streams_path' or "
                           "by setting the environment variable $STREAMSPATH.")

    # set other paths relative to top-level streams project
    _check_config_key(config, 'data_file')
    # - if it's not a full, absolute path
    if not os.path.exists(config['data_file']):
        data_file = os.path.join(config['streams_path'], config['data_file'])
        if os.path.exists(data_file):
            config['data_file'] = data_file
        else:
            raise ValueError("Invalid path to data file '{}'".format(config['data_file']))

    # set the path to write things to (for any output)
    _check_config_key(config, 'name')
    output_path = os.path.join(config['streams_path'], "plots/infer_potential/", config['name'])
    if config.has_key('output_path'):
        output_path = os.path.join(config['output_path'], config['name'])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    config['output_path'] = output_path

    # potential
    _check_config_key(config, 'potential')
    _check_config_key(config['potential'], 'class_name', more="to the 'potential' section.")
    config['potential']['parameters'] = config['potential'].get('parameters', [])

    # particles
    _check_config_key(config, 'particles')
    config['particles'] = dict() if config['particles'] is None else config['particles']
    config['particles']['parameters'] = config['particles'].get('parameters', [])

    # satellite
    _check_config_key(config, 'satellite')
    config['satellite'] = dict() if config['satellite'] is None else config['satellite']
    config['satellite']['parameters'] = config['satellite'].get('parameters', [])

    return config

def read_table(filename, expr=None, N=None):
    _table = np.genfromtxt(filename, names=True)

    if expr is not None:
        idx = numexpr.evaluate(str(expr), _table)
        _table = _table[idx]

    if N is not None and N > 0:
        idx = np.array(sample(xrange(len(_table)), min(N,len(_table))))
        _table = _table[idx]

    return _table

def read_hdf5(h5file, nparticles=None, particle_idx=None):
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

        if particle_idx is None:
            particle_idx = np.arange(0, nparticles, dtype=int)

        true_tub = ptcl["tub"].value
        true_tail_bit = ptcl["tail_bit"].value
        if "error" in ptcl.keys():
            p = ObservedParticle(ptcl["data"].value[particle_idx].T,
                                 ptcl["error"].value[particle_idx].T,
                                 frame=heliocentric,
                                 units=[u.Unit(x) for x in ptcl["units"]])

            true_p = Particle(ptcl["true_data"].value[particle_idx].T,
                              frame=heliocentric,
                              units=[u.Unit(x) for x in ptcl["units"]])
            true_p.tub = true_tub[particle_idx]
            true_p.tail_bit = true_tail_bit[particle_idx]
            return_dict["true_particles"] = true_p
        else:
            p = Particle(ptcl["data"].value.T,
                         frame=heliocentric,
                         units=[u.Unit(x) for x in ptcl["units"]])
            p.tub = true_tub
            p.tail_bit = true_tail_bit
        return_dict["particles"] = p

        if "error" in satl.keys():
            s = ObservedParticle(satl["data"].value.T, satl["error"].value.T,
                                 frame=heliocentric,
                                 units=[u.Unit(x) for x in satl["units"]])
            return_dict["true_satellite"] = Particle(satl["true_data"].value.T,
                                                     frame=heliocentric,
                                                     units=[u.Unit(x) for x in satl["units"]])
            return_dict["true_satellite"].mass = satl["m"].value
            return_dict["true_satellite"].vdisp = satl["v_disp"].value

        else:
            s = Particle(satl["data"].value.T,
                         frame=heliocentric,
                         units=[u.Unit(x) for x in satl["units"]])
        satl.mass = satl["m"].value
        s.vdisp = satl["v_disp"].value
        return_dict["satellite"] = s

        if "simulation" in f.keys():
            return_dict["t1"] = float(f["simulation"]["t1"].value)
            return_dict["t2"] = float(f["simulation"]["t2"].value)

    return return_dict