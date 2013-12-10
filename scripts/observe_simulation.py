# coding: utf-8

""" Observe particles from a given simulation """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy
import logging

# Third-party
import astropy.units as u
from astropy.utils.console import color_print
import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import yaml

# Project
from streams import usys
from streams.coordinates.frame import heliocentric, galactocentric
import streams.io as s_io
from streams.observation.gaia import gaia_spitzer_errors
import streams.potential as s_potential
from streams.util import _parse_quantity, make_path

# Create logger
logger = logging.getLogger(__name__)

def observe_simulation(class_name, error_model, missing_dims=[], selection_expr=None, N=None,
                       output_file=None, overwrite=False, seed=None, class_kwargs=dict()):
    """ Observe simulation data and write the output to a standard HDF5 format.

        TODO: handle missing dimensions

        Parameters
        ----------

    """

    if os.path.exists(output_file) and overwrite:
        os.remove(output_file)

    if os.path.exists(output_file):
        raise IOError("File '{}' already exists! Did you "
                      "want to use overwrite=True?".format(output_file))

    # read the simulation data from the specified class
    if seed is None:
        seed = np.random.randint(100)

    logger.debug("Using seed: {}".format(seed))
    np.random.seed(seed)

    try:
        Simulation = getattr(s_io, class_name)
    except AttributeError:
        raise ValueError("Simulation class '{}' not found!".format(class_name))

    # instantiate the specified simulation class with any additional keyword
    #   arguments, e.g., mass from SgrSimulation
    simulation = Simulation(**class_kwargs)

    # read particles from the simulation class
    particles = simulation.particles(N=N, expr=selection_expr)
    particles = particles.to_frame(heliocentric)

    logger.debug("Read in {} particles with expr='{}'"\
                 .format(particles.nparticles, selection_expr))

    # read the satellite position
    satellite = simulation.satellite()
    satellite = satellite.to_frame(heliocentric)
    logger.debug("Read in present position of satellite...")

    # # first get the Gaia + Spitzer errors as default
    # particle_errors = gaia_spitzer_errors(particles)
    particle_errors = error_model(particles)
    o_particles = particles.observe(particle_errors)

    error_X = np.zeros_like(o_particles._repr_X)
    for ii,n in enumerate(o_particles.frame.coord_names):
        error_X[...,ii] = o_particles.errors[n].to(o_particles._repr_units[ii]).value

    with h5py.File(output_file, "w") as f:
        # add particle positions to file
        grp = f.create_group("particles")
        grp["data"] = o_particles._repr_X
        grp["error"] = o_particles._repr_error_X
        grp["coordinate_names"] = o_particles.frame.coord_names
        grp["units"] = [str(x) for x in o_particles._repr_units]
        grp["tub"] = o_particles.tub

        grp = f.create_group("satellite")
        grp["data"] = satellite._repr_X
        grp["coordinate_names"] = satellite.frame.coord_names
        grp["units"] = [str(x) for x in satellite._repr_units]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                    default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False, action="store_true",
                        help="Overwrite any existing data.")

    parser.add_argument("-f", "--file", dest="output_file", required=True,
                        help="Name of the file to store the data.")
    parser.add_argument("--class_name", dest="class_name", required=True,
                        help="Name of the simulation data class, e.g. SgrSimulation")
    parser.add_argument("--expr", dest="expr", default=None, type=str,
                        help="Selection expression (for numexpr) for simulation particles.")
    parser.add_argument("--N", dest="N", default=None, type=int,
                        help="Number of particles.")
    parser.add_argument("--seed", dest="seed", default=None, type=int,
                        help="Seed for random number generator.")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    # TODO: missing dimensions
    # TODO: class kwargs
    observe_simulation(args.class_name, gaia_spitzer_errors, missing_dims=[],
         selection_expr=args.expr, N=args.N, output_file=args.output_file,
         overwrite=args.overwrite, seed=args.seed, class_kwargs=dict(mass="2.5e8"))
