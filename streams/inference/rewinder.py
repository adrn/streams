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
from .rewinder_likelihood import rewinder_likelihood
from .kinematicobject import KinematicObject
from ..coordinates import _hel_to_gc

__all__ = ["Rewinder", "RewinderSampler"]

def log_normal(x, mu, sigma):
    X = x - mu
    return -0.5*(np.log(2*np.pi) + 2*np.log(sigma) + (X/sigma)**2)

class Rewinder(EmceeModel):

    def __init__(self, potential, progenitor, stars, t1, t2, dt=-1., K=10):
        """ Model for tidal streams that uses backwards integration to Rewind
            the positions of stars.

            Parameters
            ----------
            potential : Potential object
            progenitor :
            stars :
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
        self.prog_hel = progenitor.values
        self.prog_err = progenitor.errors

        # Stars
        for par in stars.parameters.values():
            par.frozen = True
            self.add_parameter(par, "stars")

        self.stars = stars
        self.nstars = len(stars)

        # draw samples for each star
        self.K = K
        self.nsamples = K*self.nstars

        stars_samples_hel = np.zeros((self.nstars,self.K,6))
        self.stars_samples_lnprob = np.zeros((self.nstars,self.K))
        k = 0
        for n in range(self.nstars):
            stars_samples_hel[n] = np.random.normal(self.stars.values[n],
                                                    self.stars.errors[n],
                                                    size=(self.K,6))

            # only works for non-covariant case!
            self.stars_samples_lnprob[n] = log_normal(stars_samples_hel[n],
                                                      self.stars.values[n],
                                                      self.stars.errors[n]).sum(axis=1)

        self.stars_samples_lnprob = self.stars_samples_lnprob[np.newaxis]
        self.stars_samples_gal = _hel_to_gc(stars_samples_hel.reshape(self.nsamples,6))

        # set the tail assignments TODO: this is teh sux
        self.stars_tail = self.stars.parameters['tail'].truth

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
        for group_name,param_name,param in self._walk():
            v = parameter_values[group_name][param.name]
            ln_prior += param.prior(v)

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
            mass = parameter_values['progenitor']['m0']
        else:
            mass = parameters['progenitor']['m0'].frozen

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

        prog_gal = _hel_to_gc(prog_hel) # TODO: need better way to do this so can specify R_sun and V_circ if I want in the future

        ln_like = rewinder_likelihood(t1, t2, dt, potential,
                                      prog_gal, self.stars_samples_gal,
                                      mass, mdot, alpha, self.stars_tail)

        l = ln_like.reshape(ln_like.shape[0],self.nstars,self.K) - self.stars_samples_lnprob
        l = logsumexp(logsumexp(l,axis=2), axis=0)

        return l.sum() + log_normal(prog_hel, self.prog_hel, self.prog_err).sum()

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

        progenitor = at.Table.read(config['data_file'], path='progenitor')
        true_progenitor = at.Table.read(config['data_file'], path='true_progenitor')
        progenitor_err = at.Table.read(config['data_file'], path='error_progenitor')

        stars = at.Table.read(config['data_file'], path='stars')
        true_stars = at.Table.read(config['data_file'], path='true_stars')
        stars_err = at.Table.read(config['data_file'], path='error_stars')

        # progenitor parameter container
        # TODO: handle missing dimensions
        prog_ko = KinematicObject(hel=dict([(k,progenitor[k].data) for k in heliocentric_names]),
                          errors=dict([(k,progenitor_err[k].data) for k in heliocentric_names]),
                          truths=dict([(k,true_progenitor[k].data) for k in heliocentric_names]))

        prog_ko.parameters['m0'] = ModelParameter('m0', truth=float(progenitor['m0']),
                                                  prior=LogPrior()) # TODO: logspace?
        true_mdot = np.log(3.2*10**(np.floor(np.log10(float(progenitor['m0'])))-4))# THIS IS A HACK FOR 2.5e8!
        prog_ko.parameters['mdot'] = ModelParameter('mdot', truth=true_mdot,
                                                    prior=LogPrior()) # TODO: logspace?
        prog_ko.parameters['alpha'] = ModelParameter('alpha', shape=(1,),
                                                     prior=LogUniformPrior(1., 2.))

        # deal with extra star selection crap
        if star_idx is None:
            if nstars is 0:
                raise ValueError("If not specifying indexes, must specicy number"
                                 "of stars (nstars)")

            if expr is not None:
                expr_idx = numexpr.evaluate(expr, stars)
            else:
                expr_idx = np.ones(len(stars)).astype(bool)

            # leading tail stars
            lead_stars, = np.where((stars["tail"] == -1.) & expr_idx)
            np.random.shuffle(lead_stars)
            lead_stars = lead_stars[:nlead]

            # trailing tail stars
            trail_stars, = np.where((stars["tail"] == 1.) & expr_idx)
            np.random.shuffle(trail_stars)
            trail_stars = trail_stars[:ntrail]

            star_idx = np.append(lead_stars, trail_stars)

        stars = stars[star_idx]
        true_stars = true_stars[star_idx]
        stars_err = stars_err[star_idx]
        # TODO: handle missing dimensions
        logger.info("Running with {} stars.".format(len(stars)))
        logger.debug("Using stars: {}".format(list(star_idx)))

        star_ko = KinematicObject(hel=dict([(k,stars[k].data) for k in heliocentric_names]),
                                errors=dict([(k,stars_err[k].data) for k in heliocentric_names]),
                                truths=dict([(k,true_stars[k].data) for k in heliocentric_names]))
        star_ko.parameters['tail'] = ModelParameter('tail', truth=stars['tail'])

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

        for par in prog_ko.parameters.values():
            if par.name not in config["progenitor"]["parameters"]:
                if config["progenitor"].get("fixed_truth",False):
                    par.freeze("truth")

        for par in potential.parameters.values():
            if par.name not in config["potential"]["parameters"]:
                par.freeze("truth")

        logger.info("Drawing {} samples from each prior...".format(config["K"]))

        # Initialize the model
        model = cls(potential, prog_ko, star_ko, t1, t2, dt, K=config["K"])

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

