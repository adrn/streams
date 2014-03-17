# coding: utf-8

""" Utilities for the streams project. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import contextlib
import os, sys
import re
import logging
from datetime import datetime
import resource
import multiprocessing
from collections import OrderedDict

# Third-party
from astropy.utils import isiterable
import astropy.units as u
import numpy as np
import yaml
import yaml.constructor

__all__ = ["_validate_coord", "project_root", "streamspath", "make_path"]

# Create logger
logger = logging.getLogger(__name__)

streamspath = project_root = os.environ["STREAMSPATH"]

#
def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])

def get_memory_usage():
    """
    Returning resident size in megabytes
    """
    pid = os.getpid()
    try:
        pagesize = resource.getpagesize()
    except NameError:
        return -1024
    status_file = "/proc/%s/statm" % (pid)
    if not os.path.isfile(status_file):
        return -1024
    line = open(status_file).read()
    size, resident, share, text, library, data, dt = [int(i) for i in
line.split()]
    return resident * pagesize / (1024 * 1024) # return in megs

def _parse_quantity(q):
    try:
        val,unit = q.split()
    except AttributeError:
        val = q
        unit = u.dimensionless_unscaled

    return u.Quantity(float(val), unit)

@contextlib.contextmanager
def print_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

def get_pool(mpi=False, threads=None):
    """ Get a pool object to pass to emcee for parallel processing.
        If mpi is False and threads is None, pool is None.

        Parameters
        ----------
        mpi : bool
            Use MPI or not. If specified, ignores the threads kwarg.
        threads : int (optional)
            If mpi is False and threads is specified, use a Python
            multiprocessing pool with the specified number of threads.
    """

    if mpi:
        from emcee.utils import MPIPool

        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        logger.debug("Running with MPI...")

    elif threads > 1:
        logger.debug("Running with multiprocessing on {} cores..."\
                    .format(threads))
        pool = multiprocessing.Pool(threads)

    else:
        logger.debug("Running serial...")
        pool = None

    return pool

class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError, exc:
                raise yaml.constructor.ConstructorError('while constructing a mapping',
                    node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping
