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
from ..util import OrderedDictYAMLLoader

__all__ = ["read_config"]

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

    # stars
    _check_config_key(config, 'stars')
    config['stars'] = dict() if config['stars'] is None else config['stars']
    config['stars']['parameters'] = config['stars'].get('parameters', [])

    # progenitor
    _check_config_key(config, 'progenitor')
    config['progenitor'] = dict() if config['progenitor'] is None else config['progenitor']
    config['progenitor']['parameters'] = config['progenitor'].get('parameters', [])

    return config
