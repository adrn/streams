# coding: utf-8

""" Observe particles from a given simulation.

    Example call:

python scripts/observe_simulation.py --output="data/observed_simulation/2.5e8.hdf5" --path="data/simulation/sgr_nfw/M2.5e+08" --snapfile=SNAP113 -v -o --seed=42

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import copy
import random

# Third-party
import astropy.units as u
from astropy import log as logger
import astropy.table as at
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numexpr

# Project
from streamteam.io import SCFReader
from streams import usys, heliocentric_names, galactocentric_names
from streams.coordinates import hel_to_gal, gal_to_hel
from streams.observation.errormodels import *
from streams.potential import LM10Potential

def energy(xv):
    x = xv[:,:3].copy()
    v = xv[:,3:].copy()
    potential = LM10Potential()
    Epot = potential.evaluate(x)
    Ekin = 0.5*np.sum(v**2, axis=-1)
    return Epot + Ekin

def observe_table(tbl, error_model):
    new_tbl = tbl.copy()
    err_tbl = tbl.copy()
    err_tbl.meta = {}

    errors = error_model(tbl)
    for name in heliocentric_names:
        new_tbl[name] = np.random.normal(tbl[name], errors[name])
        err_tbl[name] = errors[name]
    return new_tbl, err_tbl

def observe_simulation(star_error_model=None, progenitor_error_model=None,
                       selection_expr=None, output_file=None, overwrite=False,
                       seed=None, simulation_path=None, snapfile=None):
    """ Observe simulation data and write the output to an HDF5 file """

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
    random.seed(seed)

    scf = SCFReader(simulation_path)
    snap_data = scf.read_snap(snapfile, units=usys)

    # select out particles that meet these cuts
    idx = numexpr.evaluate("(tub!=0)", snap_data)
    star_data = snap_data[idx]
    logger.debug("Read in {} particles".format(len(star_data)))

    # coordinate transform
    star_gc = np.vstack([star_data[n] for n in galactocentric_names]).T
    star_hel = gal_to_hel(star_gc)

    # create table for star data
    star_tbl = at.Table(star_hel, names=heliocentric_names)
    star_tbl.add_column(star_data["tub"]) # add tub

    # select bound particles to median to get satellite position
    idx = numexpr.evaluate("(tub==0)", snap_data)
    prog_data = snap_data[idx]

    # coordinate transform
    prog_gc = np.vstack([prog_data[n] for n in galactocentric_names]).T
    prog_gc = np.median(prog_gc, axis=0).reshape(1,6)
    logger.debug("Used {} particles to estimate progenitor position.".format(len(prog_data)))
    prog_hel = gal_to_hel(prog_gc)

    # create table for progenitor data
    prog_tbl = at.Table(prog_hel, names=heliocentric_names)
    prog_tbl.add_column(at.Column([snap_data["m"].sum()], name="m0")) # add mass

    # determine tail assignment for stars by relative energy
    dE = energy(star_gc) - energy(prog_gc)
    tail = np.zeros(len(star_tbl))
    lead = dE <= 0.
    trail = dE > 0.
    tail[lead] = -1. # leading tail
    tail[trail] = 1. # trailing
    star_tbl.add_column(at.Column(tail, name="tail")) # add tail

    # observe the data
    observed_star_tbl,star_err_tbl = observe_table(star_tbl, star_error_model)
    observed_prog_tbl,prog_err_tbl = observe_table(prog_tbl, progenitor_error_model)

    # make a plot of true and observed positions
    obs_hel = np.vstack([observed_star_tbl[n] for n in heliocentric_names]).T
    obs_gc = hel_to_gal(obs_hel)

    fig,axes = plt.subplots(2,2,figsize=(16,16))

    mpl = dict(markersize=3., marker='o', linestyle='none', alpha=0.5)
    axes[0,0].plot(star_gc[:,0], star_gc[:,1], **mpl)
    axes[0,1].plot(star_gc[:,0], star_gc[:,2], **mpl)
    axes[0,0].plot(obs_gc[trail,0], obs_gc[trail,1], label='trailing', c='#ca0020', **mpl)
    axes[0,1].plot(obs_gc[trail,0], obs_gc[trail,2], c='#ca0020', **mpl)
    axes[0,0].plot(obs_gc[lead,0], obs_gc[lead,1], label='leading', **mpl)
    axes[0,1].plot(obs_gc[lead,0], obs_gc[lead,2], **mpl)
    axes[0,0].legend()

    axes[1,0].plot(star_gc[:,3], star_gc[:,4], **mpl)
    axes[1,1].plot(star_gc[:,3], star_gc[:,5], **mpl)
    axes[1,0].plot(obs_gc[trail,3], obs_gc[trail,4], c='#ca0020', **mpl)
    axes[1,1].plot(obs_gc[trail,3], obs_gc[trail,5], c='#ca0020', **mpl)
    axes[1,0].plot(obs_gc[lead,3], obs_gc[lead,4], **mpl)
    axes[1,1].plot(obs_gc[lead,3], obs_gc[lead,5], **mpl)

    fname = os.path.splitext(os.path.basename(output_file))[0]
    fig.savefig(os.path.join(os.path.split(output_file)[0],
                             "{}.{}".format(fname, 'png')))

    # write tables to output_file
    observed_star_tbl.write(output_file, format="hdf5", path="stars", overwrite=overwrite)
    observed_prog_tbl.write(output_file, format="hdf5", path="progenitor", append=True)
    star_err_tbl.write(output_file, format="hdf5", path="error_stars", append=True)
    prog_err_tbl.write(output_file, format="hdf5", path="error_progenitor", append=True)
    star_tbl.write(output_file, format="hdf5", path="true_stars", append=True)
    prog_tbl.write(output_file, format="hdf5", path="true_progenitor", append=True)

    integ_tbl = at.Table(np.array([[np.nan]]))
    integ_tbl.meta['t1'] = snap_data.meta['time']
    integ_tbl.meta['t2'] = 0.
    integ_tbl.write(output_file, format="hdf5", path="integration", append=True)

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

    parser.add_argument("--output", dest="output_file", required=True,
                        help="Name of the file to store the data.")
    parser.add_argument("--expr", dest="expr", default=None, type=str,
                        help="Selection expression (for numexpr) for simulation particles.")
    parser.add_argument("--N", dest="N", default=None, type=int,
                        help="Number of particles.")
    parser.add_argument("--seed", dest="seed", default=None, type=int,
                        help="Seed for random number generator.")

    parser.add_argument("--path", dest="path", required=True, type=str,
                        help="Satellite path.")
    parser.add_argument("--snapfile", dest="snapfile", required=True, type=str,
                        help="Satellite snapfile.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    observe_simulation(star_error_model=gaia_spitzer_errors,
                       progenitor_error_model=gaia_spitzer_errors,
                       output_file=args.output_file,
                       overwrite=args.overwrite, seed=args.seed,
                       simulation_path=args.path, snapfile=args.snapfile)
