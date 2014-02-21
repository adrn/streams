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
from astropy.utils import isiterable
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
from streams.util import project_root

# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

minimum_config = """
name: test
data_file: data/observed_particles/2.5e8_N1024.hdf5
# nparticles: 4
# particle_idx: [16, 194, 575, 702]
nparticles: 8

potential:
    class_name: LawMajewski2010
{potential_params}

particles:
{particles_params}

satellite:
{satellite_params}
"""

pot_params = """
    parameters: [q1, qz, phi, v_halo]
"""

ptc_params = """
    parameters: [d]
"""

sat_params = """
    parameters: [d, mul, mub, vr]
"""

lm10_c = minimum_config.format(potential_params=pot_params,
                               particles_params="",
                               satellite_params="")

# _config = minimum_config.format(potential_params=pot_params,
#                                 particles_params=ptc_params,
#                                 satellite_params=sat_params)
_config = minimum_config.format(potential_params=pot_params,
                                particles_params="",
                                satellite_params="")

# particles_params=ptc_params,
# satellite_params=sat_params

Ncoarse = 21
Nfine = 51

output_path = os.path.join(project_root, 'plots', 'tests', 'infer_potential')
if not os.path.exists(output_path):
    os.mkdir(output_path)

def make_plot(model, idx, vals1, vals2):
    fig,axes = plt.subplots(2,2,figsize=(12,12),sharex='col')

    p = model.truths.copy()

    Ls = []
    for val in vals1:
        p[idx] = val
        Ls.append(model(p))
    axes[0,0].plot(vals1, Ls)
    axes[0,0].set_ylabel("$\ln\mathcal{L}$")
    axes[1,0].plot(vals1, np.exp(Ls-np.max(Ls)))
    axes[1,0].set_ylabel("$\mathcal{L}$")

    p = model.truths.copy()
    Ls = []
    for val in vals2:
        p[idx] = val
        Ls.append(model(p))

    #logger.debug("{} vs truth {}".format(vals[np.argmax(Ls)], truth))
    #logger.debug("{:.2f}% off".format(abs(vals[np.argmax(Ls)] - truth)/truth*100.))

    axes[0,1].set_title("zoomed")
    axes[0,1].plot(vals2, Ls)
    axes[1,1].plot(vals2, np.exp(Ls-np.max(Ls)))
    for ax in axes.flat:
        ax.axvline(model.truths[idx])

    return fig

class TestStreamModel(object):

    def setup(self):
        config = io.read_config(_config)
        self.model = si.StreamModel.from_config(config)

    def test_simple(self):

        # make sure true posterior value is higher than any randomly sampled value
        logger.debug("Checking posterior values...")
        true_ln_p = self.model.ln_posterior(self.model.truths, *self.model.lnpargs)
        true_ln_p2 = self.model(self.model.truths)
        logger.debug("\t\t At truth: {}".format(true_ln_p))

        p0 = self.model.sample_priors()
        ln_p = self.model.ln_posterior(p0, *self.model.lnpargs)
        ln_p2 = self.model(p0)
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

        test_path = os.path.join(output_path, "likelihood")
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        for jj,key in enumerate(sorted(true_potential.parameters.keys())):
            fig,axes = plt.subplots(2,1,figsize=(6,12),sharex=True)
            fig2,axes2 = plt.subplots(1,2,figsize=(12,6))

            vals = np.linspace(0.9, 1.1, 71)
            Ls = []
            for val in vals:
                p = true_potential._parameter_dict[key]
                pparams = model._given_potential_params.copy()
                pparams[key] = p*val
                potential = model._potential_class(**pparams)

                ll = si.back_integration_likelihood(model.lnpargs[0],
                                                    model.lnpargs[1],
                                                    model.lnpargs[2],
                                                    potential,
                                                    model.true_particles._X,
                                                    model.true_satellite._X,
                                                    np.log(model.true_satellite.mass),
                                                    model.true_satellite.vdisp,
                                                tail_bit=np.array([-1, -1, -1, -1, 1, 1, 1, -1]))
                Ls.append(ll)

            Ls = np.array(Ls)
            for kk in range(model.true_particles.nparticles):
                print(Ls[:,kk])
                line, = axes[0].plot(vals, Ls[:,kk], alpha=0.5,lw=2.)

                if jj == 0:
                    _X = model.true_particles.to_frame(galactocentric)._X[kk]
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

        model = self.model

        test_path = os.path.join(output_path, "model")
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        truth_dict = model._decompose_vector(model.truths)
        model.sample_priors()

        idx = 0
        for group_name,group in truth_dict.items():
            for param_name,truths in group.items():
                print(group_name, param_name)
                param = model.parameters[group_name][param_name]

                if group_name == "potential":
                    vals1 = np.linspace(param._prior.a,
                                        param._prior.b,
                                        Ncoarse)
                    vals2 = np.linspace(0.9,1.1,Nfine)*truths
                    fig = make_plot(model, idx, vals1, vals2)
                    fig.savefig(os.path.join(test_path, "{}_{}.png".format(idx,param_name)))
                    plt.close('all')
                    idx += 1

                if group_name == "particles":
                    for jj in range(param.value.shape[0]):
                        prior = model._prior_cache[("particles",param_name)]
                        truth = truths[jj]

                        mu,sigma = truth,prior.sigma[jj]
                        vals1 = np.linspace(mu-10*sigma,
                                            mu+10*sigma,
                                            Ncoarse)
                        vals2 = np.linspace(mu-3*sigma,
                                            mu+3*sigma,
                                            Nfine)
                        fig = make_plot(model, idx, vals1, vals2)
                        fig.savefig(os.path.join(test_path,
                                "ptcl{}_{}.png".format(idx,param_name)))
                        plt.close('all')
                        idx += 1

                if group_name == "satellite":

                    # if param_name == "logm0":
                    #     print(truths)
                    #     vals1 = np.linspace(param._prior.a,
                    #                         param._prior.b,
                    #                         Ncoarse)
                    #     vals2 = np.linspace(0.9,1.1,Nfine)*truths
                    #     fig = make_plot(model, idx, vals1, vals2)
                    #     fig.savefig(os.path.join(test_path, "{}_{}.png".format(idx,param_name)))
                    #     plt.close('all')
                    #     idx += 1

                    for jj in range(param.value.shape[0]):
                        prior = model._prior_cache[("satellite",param_name)]
                        truth = truths

                        mu,sigma = truth,prior.sigma
                        vals1 = np.linspace(mu-10*sigma,
                                            mu+10*sigma,
                                            Ncoarse)
                        vals2 = np.linspace(mu-3*sigma,
                                            mu+3*sigma,
                                            Nfine)
                        fig = make_plot(model, idx, vals1, vals2)
                        fig.savefig(os.path.join(test_path,
                                "sat{}_{}.png".format(idx,param_name)))
                        plt.close('all')
                        idx += 1


    def test_apw(self):
        c = io.read_config(_configs[3])
        model = si.StreamModel.from_config(c)

        p = np.array([1.1515709829358736, 1.3061651589219028, 1.6554345936276273, 0.12323310142846752, 3.0522815366139735, -0.57442121586774952, 36.751323505513533, 0.0066999310241279029, -0.0042394488370680371, -0.19422152398249989, -2.1543975690443853, 1.0121210540890484, 38.821645607755585, 0.0010389207842562375, -0.0062815802063840367, -0.009289155208715285, 5306.9338333741944, 1187.7455521689817])

        print(list(model.truths))
        print(list(model.parameters['particles']['_X']._prior[0].mu) + \
              list(model.parameters['particles']['_X']._prior[1].mu))
        print(list(model.parameters['particles']['_X']._prior[0].sigma) + \
              list(model.parameters['particles']['_X']._prior[1].sigma))

        #print(model(p))

        return

        for jj,truth in enumerate(model.truths):
            fig,axes = plt.subplots(1,2,figsize=(12,6))

            p = model.truths.copy()
            vals = truth*np.linspace(0.25, 1.75, 51)
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


def time_likelihood(model, potential, niter=100):
    for ii in range(niter):
        ll = si.back_integration_likelihood(model.lnpargs[0],
                                            model.lnpargs[1],
                                            model.lnpargs[2],
                                            potential,
                                            model.true_particles._X,
                                            model.true_satellite._X,
                                            model.true_satellite.mass,
                                            model.true_satellite.vdisp)

def time_posterior(model, niter=100):
    for ii in range(niter):
        model(model.truths)

if __name__ == "__main__":
    import cProfile
    import pstats

    c = io.read_config(lm10_c)
    model = si.StreamModel.from_config(c)
    potential = model._potential_class(**model._given_potential_params)

    cProfile.run('time_likelihood(model, potential)', 'likelihood_stats')
    p = pstats.Stats('likelihood_stats')
    p.strip_dirs().sort_stats('cumulative').print_stats(25)

    cProfile.run('time_posterior(model)', 'posterior_stats')
    p = pstats.Stats('posterior_stats')
    p.strip_dirs().sort_stats('cumulative').print_stats(25)
