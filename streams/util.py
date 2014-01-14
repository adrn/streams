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
import shutil

# Third-party
from astropy.utils.misc import isiterable
import astropy.units as u
import numpy as np

__all__ = ["_validate_coord", "project_root", "u_galactic", "make_path"]

# Create logger
logger = logging.getLogger(__name__)

# This code will find the root directory of the project
_pattr = re.compile("(.*)\/streams")
try:
    matched_path = _pattr.search(os.getcwd()).groups()[0]
except AttributeError: # match not found, try __file__ instead
    matched_path = _pattr.search(__file__).groups()[0]

if os.path.basename(matched_path) == "streams":
    project_root = matched_path
else:
    project_root = os.path.join(matched_path, "streams")

#
def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])

u_galactic = [u.kpc, u.Myr, u.M_sun, u.radian]

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

def make_path(output_data_path, name=None, overwrite=False):
    """ Make or return path for saving plots and sampler data files.

        Parameters
        ----------
        output_data_path : str
        name : str (optional)
        overwrite : bool (optional)
    """

    if name is None:
        iso_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.debug("Name not specified, using current time...")
        name = iso_now

    path = os.path.join(output_data_path, name)
    logger.debug("Output path: '{}'".format(path))

    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)

    if not os.path.exists(path):
        os.mkdir(path)

    return path

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
    # This needs to go here so I don't read in the particle file N times!!
    # get a pool object given the configuration parameters
    if mpi:
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