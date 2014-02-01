# coding: utf-8

""" Tests for infer_potential.py """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import shutil
import time

# Third-party
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import triangle

# Project
from streams.coordinates.frame import galactocentric
import streams.io as io
from streams.io.sgr import SgrSimulation
import streams.inference as si
import streams.potential as sp

# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

minimum_config = """
name: test
data_file: data/observed_particles/N1024.hdf5
nparticles: 2
particles_idx: [0, 101]

potential:
    class_name: LawMajewski2010
{potential_params}

particles:
{particles_params}

satellite:
{satellite_params}
"""

pot_params = """
    parameters:
        q1:
            a: 1.1
            b: 1.6
        qz:
            a: 1.1
            b: 1.6
        phi:
            a: 90. degree
            b: 110. degree
        v_halo:
            a: 110 km/s
            b: 140 km/s
"""

ptc_params = """
    parameters:
        _X:
        tub:
"""

sat_params = """
    parameters:
        _X:
"""

_configs = []
_configs.append(minimum_config.format(potential_params=pot_params,
                                      particles_params="",
                                      satellite_params=""))
# _configs.append(minimum_config.format(potential_params="",
#                                       particles_params=ptc_params,
#                                       satellite_params=""))
# _configs.append(minimum_config.format(potential_params="",
#                                       particles_params="",
#                                       satellite_params=sat_params))
# _configs.append(minimum_config.format(potential_params=pot_params,
#                                       particles_params=ptc_params,
#                                       satellite_params=""))
# _configs.append(minimum_config.format(potential_params=pot_params,
#                                       particles_params="",
#                                       satellite_params=sat_params))
# _configs.append(minimum_config.format(potential_params=pot_params,
#                                       particles_params=ptc_params,
#                                       satellite_params=sat_params))

class TestStreamModel(object):

    def setup(self):
        self.configs = [io.read_config(config_file) for config_file in _configs]

    def test_simple(self):

        for ii,c in enumerate(self.configs):
            print()
            logger.info("Testing config {}".format(ii+1))
            model = si.StreamModel.from_config(c)

            # make sure true posterior value is higher than any randomly sampled value
            logger.debug("Checking posterior values...")
            true_ln_p = model.ln_posterior(model.truths, *model.lnpargs)
            true_ln_p2 = model(model.truths)
            logger.debug("\t\t At truth: {}".format(true_ln_p))

            p0 = model.sample_priors()
            ln_p = model.ln_posterior(p0, *model.lnpargs)
            ln_p2 = model(p0)
            logger.debug("\t\t At random sample: {}".format(ln_p))

            assert true_ln_p > ln_p
            assert true_ln_p == true_ln_p2
            assert ln_p == ln_p2

    def test_model(self):
        """ Simple test of posterior """

        for ii,c in enumerate(self.configs):
            print()
            logger.info("Testing config {}".format(ii+1))
            model = si.StreamModel.from_config(c)

            test_path = os.path.join(c['output_path'], "config{}".format(ii))
            if not os.path.exists(test_path):
                os.mkdir(test_path)

            for jj,truth in enumerate(model.truths):
                p = model.truths.copy()
                vals = truth*np.linspace(0.9, 1.1, 51)
                Ls = []
                for val in vals:
                    p[jj] = val
                    Ls.append(model(p))

                logger.debug("{} vs truth {}".format(vals[np.argmax(Ls)], truth))
                logger.debug("{:.2f}% off".format(abs(vals[np.argmax(Ls)] - truth)/truth*100.))

                plt.clf()
                plt.plot(vals, Ls)
                plt.axvline(truth)
                plt.savefig(os.path.join(test_path, "log_test_{}.png".format(jj)))

                # plt.clf()
                # plt.plot(vals, np.exp(Ls))
                # plt.axvline(truth)
                # plt.savefig(os.path.join(test_path, "test_{}.png".format(jj)))


