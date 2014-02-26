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
from streams.dynamics import ObservedParticle
import streams.io as s_io
from streams.observation.gaia import gaia_spitzer_errors
from streams.potential.lm10 import LawMajewski2010
from streams.util import _parse_quantity
from streams.integrate import LeapfrogIntegrator

# Create logger
logger = logging.getLogger(__name__)

def observe_simulation(class_name, particle_error_model=None, satellite_error_model=None,
                       selection_expr=None, N=None, output_file=None, overwrite=False,
                       seed=None, class_kwargs=dict()):
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
    sim_time = simulation.particle_units[0]/simulation.particle_units[-1]

    # HACK HACK HACK
    selection_expr = "(tub!=0) & (tub>{}) & (tub<{}) & (sqrt((x+8)**2 + y**2 + z**2)<50)"\
                     .format((2000*u.Myr).to(sim_time).value,
                             (5200*u.Myr).to(sim_time).value)
    particles = simulation.particles(N=N, expr=selection_expr)

    logger.debug("Read in {} particles with expr='{}'"\
                 .format(particles.nparticles, selection_expr))

    # read the satellite position
    satellite = simulation.satellite()
    logger.debug("Read in present position of satellite...")

    ################################################################
    # HACK: Figure out if in leading or trailing tail
    s = satellite
    p = particles
    lm10_potential = LawMajewski2010()
    X = np.vstack((s._X[...,:3], p._X[...,:3].copy()))
    V = np.vstack((s._X[...,3:], p._X[...,3:].copy()))
    integrator = LeapfrogIntegrator(lm10_potential._acceleration_at,
                                    np.array(X), np.array(V),
                                    args=(X.shape[0], np.zeros_like(X)))
    ts, rs, vs = integrator.run(t1=simulation.t1, t2=simulation.t2, dt=-1)

    s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
    p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T
    t_idx = np.array([np.argmin(np.fabs(ts - t)) for t in p.tub])

    p_x = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
    s_x = np.array([s_orbit[jj,0] for jj in t_idx])
    diff = p_x-s_x

    norm_r = s_x[:,:3] / np.sqrt(np.sum(s_x[:,:3]**2, axis=-1))[:,np.newaxis]
    norm_diff_r = diff[:,:3] / np.sqrt(np.sum(diff[:,:3]**2, axis=-1))[:,np.newaxis]
    dot_prod_r = np.sum(norm_diff_r*norm_r, axis=-1)
    tail_bit = (dot_prod_r > 0.).astype(int)*2 - 1
    ################################################################

    satellite = satellite.to_frame(heliocentric)
    particles = particles.to_frame(heliocentric)

    # observe the particles if necessary
    if particle_error_model is not None:
        logger.info("Observing particles with {}".format(particle_error_model))
        particle_errors = particle_error_model(particles)
        o_particles = particles.observe(particle_errors)
    else:
        logger.info("Not observing particles")
        o_particles = particles

    # observe the satellite if necessary
    if satellite_error_model:
        logger.info("Observing satellite with {}".format(satellite_error_model))
        satellite_errors = satellite_error_model(satellite)
        o_satellite = satellite.observe(satellite_errors)
    else:
        logger.info("Not observing satellite")
        o_satellite = satellite

    # make a plot of true and observed positions
    true_gc_particles = particles.to_frame(galactocentric)
    gc_particles = o_particles.to_frame(galactocentric)
    all_gc_particles = simulation.particles(N=1000, expr="tub!=0")\
                                 .to_frame(galactocentric)

    fig,axes = plt.subplots(1,2,figsize=(16,8))
    markersize = 6.
    axes[0].plot(all_gc_particles["x"].value, all_gc_particles["z"].value,
                 markersize=markersize, marker='o', linestyle='none', alpha=0.25)
    axes[0].plot(true_gc_particles["x"].value, true_gc_particles["z"].value,
                 markersize=markersize, marker='o', linestyle='none', alpha=0.5)
    axes[0].plot(gc_particles["x"].value, gc_particles["z"].value,
                 markersize=markersize, marker='o', linestyle='none', alpha=0.5, c='#ca0020')

    axes[1].plot(all_gc_particles["vx"].to(u.km/u.s).value,
                 all_gc_particles["vz"].to(u.km/u.s).value,
                 markersize=markersize, marker='o', linestyle='none', alpha=0.25)
    axes[1].plot(true_gc_particles["vx"].to(u.km/u.s).value,
                 true_gc_particles["vz"].to(u.km/u.s).value,
                 markersize=markersize, marker='o', linestyle='none', alpha=0.5)
    axes[1].plot(gc_particles["vx"].to(u.km/u.s).value,
                 gc_particles["vz"].to(u.km/u.s).value,
                 markersize=markersize, marker='o', linestyle='none', alpha=0.5, c='#ca0020')

    fname = os.path.splitext(os.path.basename(output_file))[0]
    fig.savefig(os.path.join(os.path.split(output_file)[0],
                             "{}.{}".format(fname, 'png')))

    with h5py.File(output_file, "w") as f:
        # add particle positions to file
        grp = f.create_group("particles")
        grp["data"] = o_particles._repr_X

        if isinstance(o_particles, ObservedParticle):
            grp["error"] = o_particles._repr_error_X
            grp["true_data"] = particles._repr_X
        grp["coordinate_names"] = o_particles.frame.coord_names
        grp["units"] = [str(x) for x in o_particles._repr_units]
        grp["tub"] = o_particles.tub
        grp["tail_bit"] = tail_bit

        grp = f.create_group("satellite")
        grp["data"] = o_satellite._repr_X
        if isinstance(o_satellite, ObservedParticle):
            grp["error"] = o_satellite._repr_error_X
            grp["true_data"] = satellite._repr_X
        grp["coordinate_names"] = o_satellite.frame.coord_names
        grp["units"] = [str(x) for x in o_satellite._repr_units]
        grp["m"] = o_satellite.m
        grp["v_disp"] = o_satellite.v_disp

        grp = f.create_group("simulation")
        grp["t1"] = simulation.t1
        grp["t2"] = simulation.t2

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

    parser.add_argument("--mass", dest="mass", required=True, type=str,
                        help="Satellite mass.")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    """
        e.g.:

python scripts/observe_simulation.py -v --class_name=SgrSimulation --expr='tub!=0' \
--N=256 --file=/Users/adrian/Projects/streams/data/observed_particles/2.5e6_N1024.hdf5 \
--seed=42 --mass="2.5e6" --overwrite

python scripts/observe_simulation.py -v --class_name=SgrSimulation --expr='tub!=0' \
--N=256 --file=/Users/adrian/projects/streams/data/observed_particles/2.5e7_N1024.hdf5 \
--seed=42 --mass="2.5e7" --overwrite

python scripts/observe_simulation.py -v --class_name=SgrSimulation --expr='tub!=0' \
--N=256 --file=/Users/adrian/projects/streams/data/observed_particles/2.5e8_N1024.hdf5 \
--seed=42 --mass="2.5e8" --overwrite
    """

    # TODO: class kwargs
    # particle_error_model=gaia_spitzer_errors, satellite_error_model=gaia_spitzer_errors,
    # particle_error_model=gaia_spitzer_errors, satellite_error_model=None,
    observe_simulation(args.class_name,
        particle_error_model=gaia_spitzer_errors, satellite_error_model=gaia_spitzer_errors,
        selection_expr=args.expr, N=args.N, output_file=args.output_file,
        overwrite=args.overwrite, seed=args.seed, class_kwargs=dict(mass=args.mass))
