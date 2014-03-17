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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import triangle

# Project
from streams.coordinates.frame import galactocentric, heliocentric
import streams.io as io
import streams.inference as si
import streams.potential as sp
from streams.util import project_root

matplotlib.rc('lines', marker=None, linestyle='-')

# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

minimum_config = """
name: test
data_file: data/observed_particles/2.5e8_N1024_DH.hdf5
nparticles: 8
particle_idx: [255, 241, 447, 339, 213, 183, 643, 530]

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
    parameters: [p_shocked, beta, d]
"""
#    parameters: [d, mul, mub, vr]

sat_params = """
    parameters: [d, mul, mub, vr]
"""
#    parameters: [logmass, logmdot, d, mul, mub, vr]

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
Nfine = 71

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
        self.model.sample_priors()

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
                    if param_name in heliocentric.coord_names:
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

                    elif param_name == 'p_shocked':
                        for jj in range(param.value.shape[0]):
                            vals1 = np.linspace(param._prior.a[jj],
                                                param._prior.b[jj],
                                                Ncoarse)
                            vals2 = np.linspace(param._prior.a[jj],
                                                param._prior.b[jj],
                                                Nfine)

                            fig = make_plot(model, idx, vals1, vals2)
                            fig.savefig(os.path.join(test_path,
                                    "ptcl{}_{}.png".format(idx,param_name)))
                            plt.close('all')
                            idx += 1

                    elif param_name == 'beta':
                        for jj in range(param.value.shape[0]):
                            vals1 = np.linspace(param._prior.a[jj],
                                                param._prior.b[jj],
                                                Ncoarse)
                            vals2 = np.linspace(param._prior.a[jj],
                                                param._prior.b[jj],
                                                Nfine)

                            fig = make_plot(model, idx, vals1, vals2)
                            fig.savefig(os.path.join(test_path,
                                    "ptcl{}_{}.png".format(idx,param_name)))
                            plt.close('all')
                            idx += 1

                    elif param_name == 'tub':
                        for jj in range(param.value.shape[0]):
                            vals1 = np.linspace(param._prior.a[jj],
                                                param._prior.b[jj],
                                                Ncoarse)
                            vals2 = np.linspace(0.9,1.1,Nfine)*truths[jj]

                            fig = make_plot(model, idx, vals1, vals2)
                            fig.savefig(os.path.join(test_path,
                                    "ptcl{}_{}.png".format(idx,param_name)))
                            plt.close('all')
                            idx += 1

                if group_name == "satellite":

                    if param_name in heliocentric.coord_names:
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

                    elif param_name == "logmass":
                        vals1 = np.linspace(param._prior.a,
                                            param._prior.b,
                                            Ncoarse)
                        vals2 = np.linspace(0.9,1.1,Nfine)*truths
                        fig = make_plot(model, idx, vals1, vals2)
                        fig.savefig(os.path.join(test_path, "sat{}_{}.png".format(idx,param_name)))
                        plt.close('all')
                        idx += 1

                    elif param_name == "logmdot":
                        vals1 = np.linspace(param._prior.a,
                                            param._prior.b,
                                            Ncoarse)
                        vals2 = np.linspace(0.9,1.1,Nfine)*truths
                        fig = make_plot(model, idx, vals1, vals2)
                        fig.savefig(os.path.join(test_path, "sat{}_{}.png".format(idx,param_name)))
                        plt.close('all')
                        idx += 1

                    elif param_name == "alpha":
                        vals1 = np.linspace(0.5, 3.5, Ncoarse)
                        vals2 = np.linspace(0.9, 1.1, Nfine)*truths
                        fig = make_plot(model, idx, vals1, vals2)
                        fig.savefig(os.path.join(test_path, "sat{}_{}.png".format(idx,param_name)))
                        plt.close('all')
                        idx += 1

    def test_twod(self):

        x_param = 'mul'
        y_param = 'mub'

        sat_params = """
    parameters: [{}, {}]
        """.format(x_param, y_param)
        _config = minimum_config.format(potential_params="",
                                        particles_params="",
                                        satellite_params=sat_params)
        config = io.read_config(_config)
        model = si.StreamModel.from_config(config)
        model.sample_priors()
        truth_dict = model._decompose_vector(model.truths)

        nbins = 12
        x_vals = np.linspace(0.95,1.05,nbins)*truth_dict['satellite'][x_param]
        y_vals = np.linspace(0.95,1.05,nbins)*truth_dict['satellite'][y_param]

        vals = []
        X,Y = np.meshgrid(x_vals, y_vals)
        for x,y in zip(X.ravel(), Y.ravel()):
            lp = model(np.array([x,y]))
            vals.append(lp)
        vals = np.array(vals)
        vals = vals.reshape(nbins, nbins)

        plt.clf()
        plt.pcolor(X,Y,vals)
        plt.axvline(truth_dict['satellite'][x_param])
        plt.axhline(truth_dict['satellite'][y_param])
        plt.savefig(os.path.join(output_path, "log_twod_{}_{}.png".format(x_param, y_param)))

        vals = np.exp(vals-np.max(vals))
        plt.clf()
        plt.pcolor(X,Y,vals)
        plt.axvline(truth_dict['satellite'][x_param])
        plt.axhline(truth_dict['satellite'][y_param])
        plt.savefig(os.path.join(output_path, "twod_{}_{}.png".format(x_param, y_param)))

    def test_sanity(self):
        sat_params = """
    parameters: [logmass,d]
        """
        _config = minimum_config.format(potential_params="",
                                        particles_params="",
                                        satellite_params=sat_params)
        config = io.read_config(_config)
        model = si.StreamModel.from_config(config)
        model.sample_priors()
        truth_dict = model._decompose_vector(model.truths)

        v1 = model(model.truths)

        sat_params = """
    parameters: [d]
        """
        _config = minimum_config.format(potential_params="",
                                        particles_params="",
                                        satellite_params=sat_params)
        config = io.read_config(_config)
        model = si.StreamModel.from_config(config)
        model.sample_priors()
        truth_dict = model._decompose_vector(model.truths)
        v2 = model(model.truths)

        print(v1, v2)

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
