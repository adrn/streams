# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict
import logging
import os
import sys
import random

# Third-party
from astropy import log as logger
import numpy as np
from scipy.integrate import trapz
from scipy.misc import logsumexp
import gary.potential as sp
from gary.inference import EmceeModel, ModelParameter
from gary.inference.prior import *

# Project
from .. import heliocentric_names
from ..coordinates import hel_to_gal
from .likelihood import rewinder_likelihood, compute_dE
from .streamcomponent import StreamComponent, RewinderPotential

__all__ = ["Rewinder", "integrate_tub"]

def integrate_tub(ll, dt):
    nstars = ll.shape[1]
    ls = np.zeros(nstars)
    for j in range(nstars):
        A = ll[:,j].max()
        scipy_l = np.log(trapz(np.exp(ll[:,j] - A), dx=abs(dt))) + A
        ls[j] += scipy_l
    return ls

class Rewinder(EmceeModel):

    def __init__(self, rewinder_potential, progenitor, stars, dt, nsteps, extra_parameters,
                 selfgravity=True, nsamples=128):
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
        # integration
        self.dt = dt
        self.nsteps = nsteps
        self.args = (self.dt, self.nsteps)

        self.parameters = OrderedDict()

        self.rewinder_potential = rewinder_potential
        self.progenitor = progenitor
        self.stars = stars
        self.nstars = len(stars.data)
        self.selfgravity = selfgravity

        for name,p in self.rewinder_potential.parameters.items():
            logger.debug("Adding parameter {}".format(name))
            self.add_parameter(p, group='potential')

        for name,p in self.progenitor.parameters.items():
            logger.debug("Adding parameter {}".format(name))
            self.add_parameter(p, group='progenitor')

        for name,p in extra_parameters.items():
            logger.debug("Adding parameter {}".format(name))
            self.add_parameter(p, group='hyper')

        self.perfect_stars = False
        if self.stars.err is None:
            self.perfect_stars = True
            logger.warning("No uncertainties on stars")

        self.perfect_prog = False
        if self.progenitor.err is None:
            self.perfect_prog = True
            logger.warning("No uncertainties on progenitor")

        self.perfect_data = self.perfect_stars and self.perfect_prog
        if self.perfect_data:
            logger.warning("Perfect data!")

        self._ln_likelihood_tmp = np.empty((self.nsteps, self.nstars))

        if not self.perfect_prog:
            # add progenitor position as parameters
            for i,name in enumerate(heliocentric_names):
                logger.debug("Adding progenitor parameter {}".format(name))
                p = ModelParameter(name=name, prior=BasePrior())
                self.add_parameter(p, group='progenitor')

        self.nsamples = nsamples
        if not self.perfect_stars:
            tot_samples = 10000

            # draw samples for each star
            impo_samples_hel = np.zeros((self.nstars,tot_samples,6))
            impo_samples_hel[...,:2] = self.stars.data[:,np.newaxis,:2]  # copy over l,b
            impo_samples_hel[...,2:] = np.random.normal(self.stars.data[:,None,2:],
                                                        self.stars.err[:,None,2:],
                                                        size=(self.nstars,tot_samples,4)) # TODO: missing data!!??
            impo_samples = hel_to_gal(impo_samples_hel.reshape(self.nstars*tot_samples,6))

            # compute prior probabilities for the samples
            ldp = np.array([self.stars.ln_data_prob(impo_samples_hel[:,i]) for i in range(tot_samples)])
            self.impo_samples_lnprob = ldp.T

            # transform to galactocentric
            self.impo_samples_gal = impo_samples.reshape(self.nstars,tot_samples,6)

            # TODO: pre-allocate? set the tail assignments
            # tail = np.array(self.stars.parameters['tail'])

            self._ln_likelihood_tmp = np.zeros((self.nsteps, self.nsamples))

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
            if par.frozen is False and name not in heliocentric_names:
                ln_p += par.prior.logpdf(parameter_values['progenitor'][name])

        if not self.perfect_prog:
            x = np.array([[parameter_values['progenitor'][name] for name in heliocentric_names]])
            ln_p += self.progenitor.ln_data_prob(x).sum()

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
        if not self.perfect_prog:
            prog_hel = np.array([[parameter_values['progenitor'][name] for name in heliocentric_names]])
            prog_gal = hel_to_gal(prog_hel)

        else:
            prog_gal = hel_to_gal(self.progenitor.data)

        # tail assignments for each star
        tail = np.array(self.stars.parameters['tail'])

        if self.perfect_stars:
            # need to specify R_sun and V_circ as parameters
            stars_gal = hel_to_gal(self.stars.data)

            # Assume there is no uncertainty in the positions of the stars!
            rewinder_likelihood(self._ln_likelihood_tmp,
                                self.dt, self.nsteps,
                                potential.c_instance,
                                prog_gal, stars_gal,
                                m0, mdot,
                                alpha, tail, theta,
                                self.selfgravity)

            # TODO: don't create this every step
            coeffs = np.ones((self.nsteps,1))
            coeffs[0,0] = 0.5
            coeffs[self.nsteps-1,0] = 0.5
            ln_like2 = logsumexp(self._ln_likelihood_tmp, axis=0, b=coeffs) + np.log(np.fabs(dt))

            if np.any(np.isnan(ln_like2)):
                nan_ix = np.where(np.isnan(ln_like2))[0]
                return -np.inf

            likelihood = ln_like2.sum()

        else:
            likelihood = 0.

            # expected energy scale for progenitor
            a = alpha
            b = 2*alpha
            _dE = compute_dE(prog_gal, self.dt, self.nsteps, potential.c_instance, m0, mdot)

            for k in range(self.nstars):
                prog_E = potential.total_energy(prog_gal[:,:3], prog_gal[:,3:])

                # compute relative energy for each star
                star_E = potential.total_energy(self.impo_samples_gal[k,:,:3],
                                                self.impo_samples_gal[k,:,3:])
                star_dE = np.abs(star_E - prog_E)

                # only keep the samples that lie within an energy range
                Eix = np.where((star_dE > (a*_dE)) & (star_dE < (b*_dE)))[0]
                good_samples = self.impo_samples_gal[k,Eix][:self.nsamples]
                good_ln_data_prob = self.impo_samples_lnprob[k,Eix][:self.nsamples]

                # Assume there is no uncertainty in the positions of the stars!
                rewinder_likelihood(self._ln_likelihood_tmp,
                                    self.dt, self.nsteps,
                                    potential.c_instance,
                                    prog_gal, good_samples,
                                    m0, mdot,
                                    alpha, np.zeros(self.nsamples) + tail[k],
                                    theta, self.selfgravity)

                # TODO: don't create this every step
                # coeffs = np.ones((self.nsteps,1))
                # coeffs[0,0] = 0.5
                # coeffs[self.nsteps-1,0] = 0.5
                ln_like2 = logsumexp(self._ln_likelihood_tmp, axis=0) + np.log(np.fabs(dt))
                l = logsumexp(ln_like2) - np.log(self.nsamples)
                # - good_ln_data_prob
                likelihood += l

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
        star_file = config.get('star_data')
        if not os.path.exists(star_file):
            star_file = os.path.join(config['streams_path'], star_file)
        star_data = np.genfromtxt(star_file, names=True)

        prog_file = config.get('progenitor_data')
        if not os.path.exists(prog_file):
            prog_file = os.path.join(config['streams_path'], prog_file)
        prog_data = np.genfromtxt(prog_file, names=True)

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
        prog_cfg = config.get('progenitor')
        prog_obs = np.vstack([prog_data[name] for name in heliocentric_names]).T.copy()
        try:
            prog_err = np.vstack([prog_data["err_{}".format(name)] for name in heliocentric_names]).T.copy()
        except ValueError:
            logger.warning("Progenitor data uncertainty columns misnamed or don't exist.")
            prog_err = None

        # Progenitor mass:
        try:
            m0 = float(prog_cfg['mass'])
        except:
            m0 = eval(prog_cfg['mass'])

        if isinstance(m0, BasePrior):
            prior = m0
            m0 = ModelParameter(name="m0", shape=(1,), prior=prior)
        else:
            frozen = m0
            m0 = ModelParameter(name="m0", shape=(1,), prior=BasePrior())
            m0.frozen = frozen

        # Progenitor mass-loss:
        try:
            mdot = float(prog_cfg['mass_loss_rate'])
        except:
            mdot = eval(prog_cfg['mass_loss_rate'])
        if isinstance(mdot, BasePrior):
            prior = mdot
            mdot = ModelParameter(name="mdot", shape=(1,), prior=prior)
        else:
            frozen = mdot
            mdot = ModelParameter(name="mdot", shape=(1,), prior=BasePrior())
            mdot.frozen = frozen

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
        try:
            Potential = getattr(sp, config["potential"]["class"])
        except AttributeError:
            Potential = eval(config["potential"]["class"])
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
                hyperpars[name].frozen = prior

        # -------------------------------------------------------------------------------

        # Initialize the model
        model = cls(rewinder_potential=potential,
                    progenitor=prog, stars=stars,
                    dt=dt, nsteps=nintegrate,
                    extra_parameters=hyperpars)

        model.config = config

        return model
