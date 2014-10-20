# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict
import logging
import sys
import random

# Third-party
from astropy import log as logger
import numpy as np
import numexpr
import streamteam.potential as sp
from streamteam.inference import EmceeModel, ModelParameter
from streamteam.inference.prior import *

# Project
from .. import heliocentric_names
from ..coordinates import hel_to_gal
from .rewinder_likelihood import rewinder_likelihood
from .streamcomponent import StreamComponent, RewinderPotential

logger.setLevel(logging.DEBUG)

__all__ = ["Rewinder", "RewinderSampler"]

class Rewinder(EmceeModel):

    def __init__(self, rewinder_potential, progenitor, stars, dt, nsteps, extra_parameters):
        """ Model for tidal streams that uses backwards integration to Rewind
            the positions of stars.

            Parameters
            ----------
            rewinder_potential : streams.RewinderPotential
            progenitor : streams.Progenitor
            stars : streams.Stars
            dt, nsteps : float
                Integration parameters.

            Other Parameters
            ----------------
            kwargs
                Any other keyword arguments passed in are assumed to be additional
                parameters for the model.

        """
        self.parameters = OrderedDict()

        self.rewinder_potential = rewinder_potential
        self.progenitor = progenitor
        self.stars = stars
        self.nstars = len(stars.data)

        for name,p in self.rewinder_potential.parameters.items():
            logger.debug("Adding parameter {}".format(name))
            self.add_parameter(p, group='potential')

        for name,p in self.progenitor.parameters.items():
            logger.debug("Adding parameter {}".format(name))
            self.add_parameter(p, group='progenitor')

        for name,p in extra_parameters.items():
            logger.debug("Adding parameter {}".format(name))
            self.add_parameter(p, group='hyper')

        if self.stars.err is None and self.progenitor.err is None:
            logger.warning("No uncertainties on stars or progenitor -- assuming perfect data.")
            self.perfect_data = True
        else:
            self.perfect_data = False
            logger.info("Uncertainties found for stars or progenitor")

        if not self.perfect_data:
            raise NotImplementedError()

            # draw samples for each star
            self._nsamples_init = nsamples_init
            self.nsamples = nsamples

            # TODO: if any errors np.inf, sample from prior instead
            # ignore uncertainties in l,b
            stars_samples_hel = np.zeros((self.nstars, self._nsamples_init, 6))
            stars_samples_hel[...,:2] = self.stars.data[...,:2]
            stars_samples_hel[...,2:] = np.random.normal(self.stars.data[:,np.newaxis,2:],
                                                         self.stars.errors[:,np.newaxis,2:],
                                                         size=(self.nstars,self._nsamples_init,4))

            # compute prior probabilities for the samples
            self.stars_samples_lnprob = sum(self.stars.ln_prior(stars_samples_hel)).T[np.newaxis]

            # transform to galactocentric
            self.stars_samples_gal = hel_to_gal(stars_samples_hel.reshape(self.nsamples,6))

            # set the tail assignments
            # HACK: tail assignments always frozen?
            tail = np.array(self.stars.parameters['tail'].frozen)
            self.stars_samples_tail = np.repeat(tail[:,np.newaxis], self.K, axis=1)\
                                        .reshape(self.nsamples)

        # integration
        self.dt = dt
        self.nsteps = nsteps
        self.args = (self.dt, self.nsteps)

    def ln_prior(self, parameters, parameter_values, dt, nsteps):
        """ Evaluate the log-prior at the given parameter values, but
            not for the star positions, which we need in the likelihood
            function.

            Parameters
            ----------
            parameters : dict
                Dictionary of ModelParameter objects.
        """
        ln_p = 0.

        # potential prior
        ln_p += self.rewinder_potential.ln_prior(**parameter_values['potential'])

        # progenitor
        for name,par in parameters.get('progenitor',dict()).items():
            if par.frozen is False:
                ln_p += par.prior.logpdf(parameter_values['progenitor'][name])

        # hyper-parameters
        for name,par in parameters.get('hyper',dict()).items():
            if par.frozen is False:
                ln_p += par.prior.logpdf(parameter_values['hyper'][name])

        return ln_p

    def ln_likelihood(self, parameters, parameter_values, dt, nsteps):
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

        #################################################################################
        # potential parameters:
        #
        potential = self.rewinder_potential.obj(**parameter_values['potential'])

        #################################################################################
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

        #################################################################################
        # Hyper parameters:
        #

        # scale of position of effective tidal radius
        if parameters['hyper']['alpha'].frozen is False:
            alpha = parameter_values['hyper']['alpha']
        else:
            alpha = parameters['hyper']['alpha'].frozen

        # rotation of Lagrange points in orbital plane
        if parameters['hyper']['theta'].frozen is False:
            theta = parameter_values['hyper']['theta']
        else:
            theta = parameters['hyper']['theta'].frozen

        # -------------------------------------------------------------------------------
        # satellite coordinates
        #

        # TODO: assume perfect knowledge of stars, but missing dims for progenitor!
        if self.perfect_data:
            # need to specify R_sun and V_circ as parameters
            prog_gal = hel_to_gal(self.progenitor.data)
            stars_gal = hel_to_gal(self.stars.data)
            tail = np.array(self.stars.parameters['tail'])

            # Assume there is no uncertainty in the positions of the stars!
            ln_like = rewinder_likelihood(self.dt, self.nsteps,
                                          potential.c_instance,
                                          prog_gal, stars_gal,
                                          m0, mdot,
                                          alpha, tail, theta)

            likelihood = ln_like.sum()

        else:
            raise NotImplementedError()

            # TODO: need to handle progenitor position as parameters
            prog_hel = np.empty((1,6))
            for i,par_name in enumerate(heliocentric_names):
                if parameters['progenitor'][par_name].frozen is False:
                    prog_hel[:,i] = parameter_values['progenitor'][par_name]
                else:
                    prog_hel[:,i] = parameters['progenitor'][par_name].frozen

            # Thin the samples of stars so that they lie within some sensible range of
            #   energies relative to the progenitor
            prog_E = potential.energy(prog_gal[:,:3], prog_gal[:,3:])
            samples_E = potential.energy(self.stars_samples_gal[:,:3], self.stars_samples_gal[:,3:])

            # compute back-integration likelihood for all samples
            ln_like = rewinder_likelihood(t1, t2, dt, potential,
                                          prog_gal, self.stars_samples_gal,
                                          m0, mdot, alpha, self.stars_samples_tail)
            ln_like = ln_like.reshape(ln_like.shape[0],self.nstars,self.K)

        return likelihood

    def sample_p0(self, n=None):
        """ """

        p0 = np.zeros((n, self.nparameters))
        ix1 = 0
        for group_name,param_name,param in self._walk():
            if group_name == 'progenitor' and param_name in heliocentric_names:
                if param.size == 1:
                    size = (n,)
                else:
                    size = (n,)+self.progenitor.parameters[param_name].shape
                samples = np.random.normal(self.progenitor.parameters[param_name].value,
                                           self.progenitor.parameters[param_name].error,
                                           size=size)
            else:
                samples = param.prior.sample(n=n)

            if param.size == 1:
                p0[:,ix1] = samples
            else:
                p0[:,ix1:ix1+param.size] = samples

            ix1 += param.size

        return p0


    # -----------------------------------------------------------------------------------

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

        # Set the log level based on config file - default is debug
        log_level = config.get('log_level', "DEBUG")
        logger.setLevel(getattr(logging, log_level.upper()))

        # Use a seed for random number generators
        seed = config.get('seed', np.random.randint(100000))
        logger.debug("Using seed: {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)

        # Read star data from specified file
        star_data = np.genfromtxt(config.get('star_data'), names=True)
        prog_data = np.genfromtxt(config.get('progenitor_data'), names=True)

        # If limiting the number of stars
        # TODO: allow selecting on some expression
        try:
            nstars = int(config.get('nstars'))
        except TypeError:
            nstars = None

        if nstars is not None:
            star_data = star_data[:nstars]
        logger.info("Using {} stars".format(len(star_data)))

        # Turn star and progenitor tables into 6D arrays with proper structure
        stars_obs = np.vstack([star_data[name] for name in heliocentric_names]).T.copy()
        try:
            stars_err = np.vstack([star_data["err_{}".format(name)] for name in heliocentric_names]).T.copy()
        except ValueError:
            logger.warning("Star data uncertainty columns misnamed or don't exist.")
            stars_err = None

        stars = StreamComponent(stars_obs, err=stars_err,
                                parameters=OrderedDict([('tail',star_data['tail'])]))

        # -------------------------------------------------------------------------------
        prog_obs = np.vstack([prog_data[name] for name in heliocentric_names]).T.copy()
        try:
            prog_err = np.vstack([prog_data["err_{}".format(name)] for name in heliocentric_names]).T.copy()
        except ValueError:
            logger.warning("Progenitor data uncertainty columns misnamed or don't exist.")
            prog_err = None

        # Progenitor mass:
        m0 = np.nan
        try:
            m0 = prog_data['mass']
        except ValueError:
            logger.info("No progenitor mass measurement in data file. Assuming a mass range "
                        "has been provided in the config (.yml) file.")

        # no mass provided
        if np.isnan(m0):
            try:
                mass_range = map(float, config.get('progenitor').get('mass_range'))
            except:
                logger.error("Failed to get mass range from config file! Aborting...")
                sys.exit(1)

            prior = LogarithmicPrior(*mass_range)
            m0 = ModelParameter(name="m0", shape=(1,), prior=prior)
        else:
            m0 = ModelParameter(name="m0", shape=(1,), prior=BasePrior())
            m0.fixed = prog_data['mass']

        # Progenitor mass-loss
        # TODO: this is too rigid...
        try:
            mass_loss = float(config.get('progenitor').get('mass_loss'))
        except:
            logger.warning("Failed to get mass-loss rate from config file! Assuming no mass-loss.")
            mass_loss = 0.

        # TODO: prior
        prior = BasePrior()
        mdot = ModelParameter(name="mdot", shape=(1,), prior=prior)
        mdot.fixed = mass_loss

        prog = StreamComponent(prog_obs, err=prog_err,
                               parameters=OrderedDict([('m0',m0), ('mdot',mdot)]))

        # -------------------------------------------------------------------------------

        # Read integration stuff
        dt = float(config['integration'].get('dt'))
        nintegrate = int(config['integration'].get('nsteps'))
        logger.debug("Will integrate for {} steps, with a timestep of {} Myr"
                     .format(nintegrate, dt))
        logger.info("Integration time: {} Myr".format(nintegrate*dt))

        # -------------------------------------------------------------------------------

        # Potential
        Potential = getattr(sp, config["potential"]["class"])
        logger.info("Using potential '{}'...".format(Potential))

        # potential parameters to vary
        vary_pars = config["potential"].get("parameters", dict())
        for k,v in vary_pars.items():
            vary_pars[k] = eval(v)

        # get fixed parameters
        fixed_pars = config["potential"].get("fixed", dict())

        potential = RewinderPotential(Potential, priors=vary_pars, fixed_pars=fixed_pars)

        # -------------------------------------------------------------------------------

        # Hyper-parameters
        hyperpars = config.get('hyperparameters', dict())
        for name,v in hyperpars.items():
            logger.debug("Adding hyper-parameter: {}".format(name))
            try:
                prior = float(v)
                logger.debug("--- {} = {}".format(name,prior))
            except ValueError:
                prior = eval(v)
                logger.debug("--- prior passed in for {}: {}".format(name,prior))

            if isinstance(prior, BasePrior):
                hyperpars[name] = ModelParameter(name=name, prior=prior)
            else:
                hyperpars[name] = ModelParameter(name=name, prior=BasePrior())
                hyperpars[name].fixed = prior

        # -------------------------------------------------------------------------------

        # Initialize the model
        model = cls(rewinder_potential=potential,
                    progenitor=prog, stars=stars,
                    dt=dt, nsteps=nintegrate,
                    extra_parameters=hyperpars)

        return model
