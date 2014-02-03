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
nparticles: 4
particle_idx: [0, 1, 512, 513]

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
            a: 1.
            b: 1.7
        qz:
            a: 1.
            b: 1.7
        phi:
            a: 80. degree
            b: 120. degree
        v_halo:
            a: 100 km/s
            b: 200 km/s
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

lm10_c = minimum_config.format(potential_params=pot_params,
                               particles_params="",
                               satellite_params="")

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

    def test_likelihood(self):

        simulation = SgrSimulation("2.5e8")
        all_gc_particles = simulation.particles(N=1000, expr="tub!=0")\
                                     .to_frame(galactocentric)

        c = io.read_config(lm10_c)
        model = si.StreamModel.from_config(c)
        true_potential = model._potential_class(**model._given_potential_params)

        test_path = os.path.join(c['output_path'], "lm10")
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        for jj,key in enumerate(sorted(true_potential.parameters.keys())):
            fig,axes = plt.subplots(2,1,figsize=(6,12),sharex=True)
            fig2,axes2 = plt.subplots(1,2,figsize=(12,6))

            vals = np.linspace(0.9, 1.1, 71)
            Ls = []
            for val in vals:
                p = true_potential.parameters[key]
                pparams = model._given_potential_params.copy()
                pparams[key] = p._value*val
                potential = model._potential_class(**pparams)

                ll = si.back_integration_likelihood(model.lnpargs[0],
                                                    model.lnpargs[1],
                                                    model.lnpargs[2],
                                                    potential, model.true_particles._X,
                                                    model.true_satellite._X,
                                                    model.true_particles.tub)
                Ls.append(ll)

            Ls = np.array(Ls)
            for kk in range(model.true_particles.nparticles):
                line, = axes[0].plot(vals, Ls[:,kk], alpha=0.5,lw=2.)
                _X = model.true_particles.to_frame(galactocentric)._X[kk]

                if jj == 0:
                    if kk == 0:
                        axes2[0].plot(all_gc_particles._X[:,0], all_gc_particles._X[:,2],
                                      marker='.',alpha=0.1, color='k',linestyle='none',ms=5)
                        axes2[1].plot(all_gc_particles._X[:,3], all_gc_particles._X[:,5],
                                      marker='.',alpha=0.1, color='k',linestyle='none',ms=5)

                    axes2[0].plot(_X[0],_X[2],marker='o',alpha=1.,
                                  color=line.get_color(),ms=8,
                                  label="{}, tub={}".format(kk,model.true_particles.tub[kk]))
                    axes2[1].plot(_X[3],_X[5],marker='o',alpha=1.,color=line.get_color(),ms=8)

            if jj == 0:
                axes2[0].legend(loc='lower left', fontsize=8)
                axes2[0].set_xlabel("X")
                axes2[0].set_ylabel("Z")
                axes2[1].set_xlabel("vx")
                axes2[1].set_ylabel("vz")
                fig2.savefig(os.path.join(test_path, "particles.png"))

            axes[1].plot(vals, np.sum(Ls,axis=1), alpha=0.75,lw=2.,color='k')
            fig.suptitle(key)
            fig.savefig(os.path.join(test_path, "log_test_{}.png".format(jj)))

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
                fig,axes = plt.subplots(1,2,figsize=(12,6))

                p = model.truths.copy()
                vals = truth*np.linspace(0.25, 1.75, 121)
                Ls = []
                for val in vals:
                    p[jj] = val
                    Ls.append(model(p))
                axes[0].plot(vals, Ls)

                p = model.truths.copy()
                vals = truth*np.linspace(0.9, 1.1, 121)
                Ls = []
                for val in vals:
                    p[jj] = val
                    Ls.append(model(p))

                logger.debug("{} vs truth {}".format(vals[np.argmax(Ls)], truth))
                logger.debug("{:.2f}% off".format(abs(vals[np.argmax(Ls)] - truth)/truth*100.))

                axes[1].plot(vals, Ls)
                axes[0].axvline(truth)
                axes[1].axvline(truth)

                fig.savefig(os.path.join(test_path, "log_test_{}.png".format(jj)))

                # plt.clf()
                # plt.plot(vals, np.exp(Ls))
                # plt.axvline(truth)
                # plt.savefig(os.path.join(test_path, "test_{}.png".format(jj)))


