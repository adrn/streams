# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from collections import OrderedDict
import time
import random

# Third-party
from astropy import log as logger
import astropy.table as at
from emcee import EnsembleSampler, PTSampler
import numpy as np
import astropy.units as u
import h5py
import numexpr
from scipy.stats import norm
from scipy.misc import logsumexp

# Project
from streamteam.inference import EmceeModel, ModelParameter, LogUniformPrior, LogPrior
from .. import heliocentric_names
from ..util import streamspath
from ..coordinates import hel_to_gal
from .rewinder_likelihood import rewinder_likelihood
from .kinematicobject import Stars, Progenitor
from .util import log_normal

__all__ = ["Rewinder", "RewinderSampler"]

class Rewinder(EmceeModel):

    def __init__(self, potential, progenitor, stars, t1, t2, dt=-1., K=10):
        """ Model for tidal streams that uses backwards integration to Rewind
            the positions of stars.

            Parameters
            ----------
            potential : streams.Potential
            progenitor : streams.Progenitor
            stars : streams.Stars
            t1,t2,dt : float
                Integration parameters.
            K : int
                Number of samples to draw for importance sampling.
        """

        self.parameters = OrderedDict()

        # Potential
        for par in potential.parameters.values():
            self.add_parameter(par, "potential")
        self.potential_class = potential.c_class

        # Progenitor
        for par in progenitor.parameters.values():
            self.add_parameter(par, "progenitor")
        self.progenitor = progenitor

        # Stars
        for par in stars.parameters.values():
            par.frozen = True
            self.add_parameter(par, "stars")

        self.stars = stars
        self.nstars = len(stars)

        # draw samples for each star
        self.K = K
        self.nsamples = K*self.nstars

        # TODO: if any errors np.inf, sample from prior instead
        stars_samples_hel = np.random.normal(self.stars.data[:,np.newaxis],
                                             self.stars.errors[:,np.newaxis],
                                             size=(self.nstars,self.K,6))

        # compute prior probabilities for the samples
        self.stars_samples_lnprob = self.stars.ln_prior(stars_samples_hel).T[np.newaxis]

        # transform to galactocentric
        self.stars_samples_gal = hel_to_gal(stars_samples_hel.reshape(self.nsamples,6))

        # set the tail assignments
        # HACK: tail assignments always frozen?
        tail = np.array(self.stars.parameters['tail'].frozen)
        self.stars_samples_tail = np.repeat(tail[:,np.newaxis], self.K, axis=1)\
                                    .reshape(self.nsamples)

        # integration
        self.args = (t1,t2,dt)

    def ln_prior(self, parameters, parameter_values, t1, t2, dt):
        """ Evaluate the log-prior at the given parameter values, but
            not for the star positions, which we need in the likelihood
            function.

            Parameters
            ----------
            parameters : dict
                Dictionary of ModelParameter objects.
            parameter_values : dict
                The dictionary of model parameter values.
            t1,t2,dt : numeric
                Integration limits.
        """
        ln_prior = 0.

        for param in parameters['potential'].values():
            if param.frozen is False:
                v = parameter_values['potential'][param.name]
                ln_prior += param.prior(v)

        if parameters['progenitor']['l'].frozen is False:
            # this is actually the data likelihood
            ln_prior += log_normal(parameter_values['progenitor']['l'],
                                   self.progenitor.data[:,0],
                                   self.progenitor.errors[:,0])

        if parameters['progenitor']['b'].frozen is False:
            # uniform angles over the sky
            ln_prior += np.log(np.cos(b) / (4*np.pi))

            # this is actually the data likelihood
            ln_prior += log_normal(parameter_values['progenitor']['b'],
                                   self.progenitor.data[:,1],
                                   self.progenitor.errors[:,1])

        if parameters['progenitor']['d'].frozen is False:
            # distance prior from 1 kpc to 200 kpc
            d = parameter_values['d']
            ln_prior += np.log((1 / np.log(200./1.))) - np.log(d)

            # this is actually the data likelihood
            ln_prior += log_normal(parameter_values['progenitor']['d'],
                                   self.progenitor.data[:,2],
                                   self.progenitor.errors[:,2])
        else:
            d = parameters['progenitor']['d'].frozen

        if parameters['progenitor']['mul'].frozen is False:
            ln_prior += log_normal(mul, 0., 0.306814 / d) # 300 km/s

            # this is actually the data likelihood
            ln_prior += log_normal(parameter_values['progenitor']['mul'],
                                   self.progenitor.data[:,3],
                                   self.progenitor.errors[:,3])

        if parameters['progenitor']['mub'].frozen is False:
            ln_prior += log_normal(mub, 0., 0.306814 / d) # 300 km/s

            # this is actually the data likelihood
            ln_prior += log_normal(parameter_values['progenitor']['mub'],
                                   self.progenitor.data[:,4],
                                   self.progenitor.errors[:,4])

        if parameters['progenitor']['vr'].frozen is False:
            ln_prior += log_normal(vr, 0., 0.306814) # 300 km/s

            # this is actually the data likelihood
            ln_prior += log_normal(parameter_values['progenitor']['vr'],
                                   self.progenitor.data[:,5],
                                   self.progenitor.errors[:,5])

        return ln_prior

    def ln_likelihood(self, parameters, parameter_values, t1, t2, dt):
        """ Evaluate the log-likelihood at the given parameter values.

            Parameters
            ----------
            parameters : dict
                Dictionary of ModelParameter objects.
            parameter_values : dict
                The dictionary of model parameter values.
            t1,t2,dt : numeric
                Integration limits.
        """

        ######################################################################
        # potential parameters:
        #
        pparams = dict()
        for par in parameters['potential'].values():
            # parameter is free to vary, take value from the dictionary of
            #   variable parameter values
            if par.frozen is False:
                val = parameter_values['potential'][par.name]

            # parameter is frozen to some value - take value from what it is
            #   frozen to
            else:
                val = par.frozen

            pparams[par.name] = val
        potential = self.potential_class(**pparams)

        ######################################################################
        # progenitor parameters:
        #

        # mass
        if parameters['progenitor']['m0'].frozen is False:
            m0 = parameter_values['progenitor']['m0']
        else:
            m0 = parameters['progenitor']['m0'].frozen

        # mass-loss
        if parameters['progenitor']['mdot'].frozen is False:
            mdot = parameter_values['progenitor']['mdot']
        else:
            mdot = parameters['progenitor']['mdot'].frozen

        # position of effective tidal radius
        if parameters['progenitor']['alpha'].frozen is False:
            alpha = parameter_values['progenitor']['alpha']
        else:
            alpha = parameters['progenitor']['alpha'].frozen

        # satellite coordinates
        prog_hel = np.empty((1,6))
        for i,par_name in enumerate(heliocentric_names):
            if parameters['progenitor'][par_name].frozen is False:
                prog_hel[:,i] = parameter_values['progenitor'][par_name]
            else:
                prog_hel[:,i] = parameters['progenitor'][par_name].frozen

        # TODO: need to specify R_sun and V_circ as parameters?
        prog_gal = hel_to_gal(prog_hel)

        # compute back-integration likelihood for all samples
        ln_like = rewinder_likelihood(t1, t2, dt, potential,
                                      prog_gal, self.stars_samples_gal,
                                      m0, mdot, alpha, self.stars_samples_tail)
        ln_like = ln_like.reshape(ln_like.shape[0],self.nstars,self.K)

        marg = logsumexp(ln_like, axis=0) + np.log(abs(dt))
        ln_q_jk = marg - self.stars_samples_lnprob
        n_eff = np.exp(2*logsumexp(ln_q_jk, axis=-1) - logsumexp(2*ln_q_jk, axis=-1))
        print(n_eff)

        return logsumexp(ln_q_jk - np.log(self.K), axis=-1).sum()

        # ln_q_jk = ln_like - self.stars_samples_lnprob
        # n_eff = np.exp(2*logsumexp(ln_q_jk, axis=2) - logsumexp(2*ln_q_jk, axis=2))

        # import matplotlib.pyplot as plt
        # for n in range(self.nstars):
        #     fig,axes = plt.subplots(2,1,sharex=True)
        #     fig.suptitle(r"Star {}".format(n), fontsize=24)
        #     t = np.linspace(self.args[0], self.args[1], len(n_eff[:,n]))
        #     axes[0].plot(t, n_eff[:,n], marker=None)
        #     axes[0].axvline(self.stars.parameters['tub'].truth.value[n],
        #                     linestyle='dashed', c='g', alpha=0.6)
        #     axes[0].set_ylabel(r"$N_{\rm eff}$")
        #     axes[0].set_xlim(0.,self.args[0])
        #     axes[0].set_ylim(0.,self.K)
        #     axes[1].plot(t, ln_q_jk[:,n], marker=None, alpha=0.1)
        #     axes[1].axvline(self.stars.parameters['tub'].truth.value[n],
        #                     linestyle='dashed', c='g', alpha=0.6)
        #     axes[1].set_ylabel(r"$\ln q_k(\tau)$")
        #     axes[-1].set_xlabel(r"Release time, $\tau$")
        #     fig.savefig("/Users/adrian/Downloads/star_{}.png".format(n))

        # sys.exit(0)

        return l.sum()

    # ========================================================================
    # ========================================================================
    # ========================================================================

    @classmethod
    def from_config(cls, config):
        """ Construct a StreamModel from a configuration dictionary.
            Typically comes from a YAML file via `streams.io.read_config`.

            Parameters
            ----------
            config : dict
        """

        if not isinstance(config, dict):
            from ..io import read_config
            config = read_config(config)

        seed = config.get('seed', np.random.randint(100000))
        logger.debug("Using seed: {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)

        nlead = config.get('nlead', 0)
        ntrail = config.get('ntrail', 0)
        nstars = nlead + ntrail

        # more complicated selections
        star_idx = config.get('star_idx', None)
        expr = config.get('expr', None)

        if star_idx is not None and nstars is not 0 and len(star_idx) != 0:
            raise ValueError("Number of stars does not match length of"
                             "star index specification.")

        # load progenitor and star data
        logger.debug("Reading data from:\n\t{}".format(config['data_file']))

        progenitor_data = at.Table.read(config['data_file'], path='progenitor')
        try:
            true_progenitor = at.Table.read(config['data_file'], path='true_progenitor')
        except:
            true_progenitor = None
        progenitor_err = at.Table.read(config['data_file'], path='error_progenitor')

        stars_data = at.Table.read(config['data_file'], path='stars')
        try:
            true_stars = at.Table.read(config['data_file'], path='true_stars')
        except:
            true_stars = None
        stars_err = at.Table.read(config['data_file'], path='error_stars')

        # progenitor object
        progenitor = Progenitor(data=progenitor_data,
                                errors=progenitor_err,
                                truths=true_progenitor)
        progenitor.parameters['m0'] = ModelParameter('m0',
                                                     truth=float(progenitor_data['m0']),
                                                     prior=LogPrior()) # TODO: logspace?

        # HACK: THIS IS A HACK FOR SGR SIMS!
        true_mdot = np.log(3.2*10**(np.floor(np.log10(float(progenitor_data['m0'])))-4))
        progenitor.parameters['mdot'] = ModelParameter('mdot', truth=true_mdot, prior=LogPrior())

        progenitor.parameters['alpha'] = ModelParameter('alpha', shape=(1,),
                                                        prior=LogUniformPrior(1., 2.))

        # deal with extra star selection crap
        if star_idx is None:
            if nstars is 0:
                raise ValueError("If not specifying indexes, must specify number"
                                 "of stars (nstars)")

            if expr is not None:
                expr_idx = numexpr.evaluate(expr, stars_data)
            else:
                expr_idx = np.ones(len(stars_data)).astype(bool)

            # leading tail stars
            lead_stars, = np.where((stars_data["tail"] == -1.) & expr_idx)
            np.random.shuffle(lead_stars)
            lead_stars = lead_stars[:nlead]

            # trailing tail stars
            trail_stars, = np.where((stars_data["tail"] == 1.) & expr_idx)
            np.random.shuffle(trail_stars)
            trail_stars = trail_stars[:ntrail]

            star_idx = np.append(lead_stars, trail_stars)

        stars_data = stars_data[star_idx]
        true_stars = true_stars[star_idx]
        stars_err = stars_err[star_idx]

        logger.info("Running with {} stars.".format(len(stars_data)))
        logger.debug("Using stars: {}".format(list(star_idx)))

        stars = Stars(data=stars_data, errors=stars_err, truths=true_stars)
        stars.parameters['tail'] = ModelParameter('tail', truth=stars_data['tail'].data)
        stars.parameters['tub'] = ModelParameter('tub', truth=stars_data['tub'].data)

        # integration stuff
        integration = at.Table.read(config['data_file'], path='integration')
        t1 = config.get("t1", integration.meta['t1'])
        t2 = config.get("t2", integration.meta['t2'])
        dt = config.get("dt", -1.)
        logger.debug("Integration from {} to {}, ∆t={} Myr".format(t1,t2,dt))

        # get the potential object specified from the potential subpackage
        from .. import potential as sp
        Potential = getattr(sp, config["potential"]["class_name"])
        potential = Potential()
        logger.info("Using potential '{}'...".format(config["potential"]["class_name"]))

        for par in progenitor.parameters.values():
            if par.name not in config["progenitor"]["parameters"]:
                par.freeze("truth")

        for par in potential.parameters.values():
            if par.name not in config["potential"]["parameters"]:
                par.freeze("truth")

        logger.info("Drawing {} samples from each prior...".format(config["K"]))

        # Initialize the model
        model = cls(potential, progenitor, stars, t1, t2, dt, K=config["K"])

        return model

    def _walk(self):
        for tup in super(Rewinder,self)._walk():
            if tup[0] == 'stars':
                continue
            yield tup

# TODO: check here down

class RewinderSampler(EnsembleSampler):

    def __init__(self, model, nwalkers=None, pool=None, a=2.):
        """ """

        if nwalkers is None:
            nwalkers = model.nparameters*2 + 2
        self.nwalkers = nwalkers

        super(StreamModelSampler, self).__init__(self.nwalkers, model.nparameters, model,
                                                 pool=pool, a=a)

    def write(self, filename, ii=None):
        if ii is None:
            ii = self.chain.shape[1]

        # write the sampler data to an HDF5 file
        logger.info("Writing sampler data to '{}'...".format(filename))
        with h5py.File(filename, "w") as f:
            f["last_step"] = ii
            f["chain"] = self.chain
            f["lnprobability"] = self.lnprobability
            f["acceptance_fraction"] = self.acceptance_fraction
            try:
                f["acor"] = self.acor
            except:
                logger.warn("Failed to compute autocorrelation time.")
                f["acor"] = []

    def run_inference(self, pos, nsteps, path, output_every=None,
                      output_file_fmt="inference_{:06d}.hdf5", first_step=0):
        """ Custom run MCMC that caches the sampler every specified number
            of steps.
        """
        if output_every is None:
            output_every = nsteps

        logger.info("Running {} walkers for {} steps..."\
                .format(self.nwalkers, nsteps))

        time0 = time.time()
        ii = first_step
        for outer_loop in range(nsteps // output_every):
            self.reset()
            for results in self.sample(pos, iterations=output_every):
                ii += 1

            self.write(os.path.join(path,output_file_fmt.format(ii)), ii=ii)
            pos = results[0]

        # the remainder
        remainder = nsteps % output_every
        if remainder > 0:
            self.reset()
            for results in self.sample(pos, iterations=remainder):
                ii += 1

            self.write(os.path.join(path,output_file_fmt.format(ii)), ii=ii)

        t = time.time() - time0
        logger.debug("Spent {} seconds on main sampling...".format(t))

