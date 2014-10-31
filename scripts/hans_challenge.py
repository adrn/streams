# coding: utf-8

""" Gaia Challenge 2 -- Pal 5 Challenge """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging
import os
import sys

# Third-party
from astropy import log as logger
import matplotlib.pyplot as plt
import numpy as np
import streamteam.dynamics as sd
import streamteam.integrate as si
import streamteam.io as io
import streamteam.potential as sp
from streamteam.units import galactic
from streamteam.util import get_pool
import triangle

from astropy.constants import G
Gee = G.decompose(galactic).value

# streams
from streams.util import streamspath
from streams.rewinder import Rewinder, RewinderSampler
import streams.coordinates as stc

global pool
pool = None

def plot_traces(chain, p0=None, truths=None):
    figs = []
    for i in range(chain.shape[-1]):
        fig,ax = plt.subplots(1,1,figsize=(10,6))
        for ch in chain[...,i]:
            ax.plot(ch, marker=None, drawstyle='steps', alpha=0.2, color='k')

        if p0 is not None:
            for pp in p0[:,i]:
                ax.axhline(pp, alpha=0.2, color='r')

        if truths is not None:
            ax.axhline(truths[i], alpha=0.7, color='g')

        figs.append(fig)

    return figs

def main(ix, mpi=False, overwrite=False, evol=False):
    pool = get_pool(mpi=mpi)

    if evol:
        stat_evol = "evol"
    else:
        stat_evol = "stat"

    cfg_path = os.path.join(streamspath, "config/hans_challenge{}_{}.yml".format(ix, stat_evol))
    logger.debug(cfg_path)
    model = Rewinder.from_config(cfg_path)

    out_path = os.path.join(streamspath, "output/{}".format(model.config['name']))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    chain_file = os.path.join(out_path, "chain.npy")

    sampler = RewinderSampler(model, nwalkers=64, pool=pool)

    true_parameter_values = dict(potential=dict(v_h=1., r_h=12),
                                 progenitor=dict(m0=2E9),
                                 hyper=dict(alpha=1.125, theta=0.))

    truth = model.vectorize(true_parameter_values)

    if overwrite and os.path.exists(chain_file):
        os.remove(chain_file)

    if not os.path.exists(chain_file):
        # p0_sigma = model.vectorize(parameter_sigmas)
        p0_sigma = np.abs(truth*1E-6)
        p0 = np.random.normal(truth, p0_sigma, size=(sampler.nwalkers, sampler.dim))

        # burn in
        sampler.run_inference(p0, 100)
        best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]

        figs = plot_traces(sampler.chain, p0=None, truths=truth)
        for i,fig in enumerate(figs):
            fig.savefig(os.path.join(out_path, "burn_{}.png".format(i)))

        sampler.reset()
        logger.info("Done burning in")

        # restart walkers from best position, burn again
        new_pos = np.random.normal(best_pos, p0_sigma,
                                   size=(sampler.nwalkers, p0.shape[1]))
        sampler.run_inference(new_pos, 500)
        pos = sampler.chain[:,-1].copy()
        sampler.reset()

        logger.info("Done re-burn")

        # run for inference steps
        sampler.run_inference(pos, 500)

        logger.debug("Acceptance fraction: {}".format(sampler.acceptance_fraction))

        chain = sampler.chain
        np.save(chain_file, chain)
    else:
        chain = np.load(chain_file)

    figs = plot_traces(chain, p0=None, truths=truth)
    for i,fig in enumerate(figs):
        fig.savefig(os.path.join(out_path, "{}.png".format(i)))

    flatchain = np.vstack(chain)
    extents = [(0.8,1.2), (5,30)]
    fig = triangle.corner(flatchain, truths=truth)
    #                      extents=extents)
    #                      labels=[r"$M$ [$M_\odot$]", r"$R_h$ [kpc]", "$q_z$"])
    fig.savefig(os.path.join(out_path, "corner.png"))

    pool.close()
    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False, 
                        action="store_true", help="Nukem.")
    parser.add_argument("-e", "--evol", dest="evol", default=False, 
                        action="store_true", help="Evolving instead of static")

    # threading
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")

    parser.add_argument("-i", dest="ix", required=True)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        main(args.ix, mpi=args.mpi, overwrite=args.overwrite, evol=args.evol)
    except:
        pool.close() if hasattr(pool, 'close') else None
        raise
        sys.exit(1)

    pool.close() if hasattr(pool, 'close') else None
    sys.exit(0)
