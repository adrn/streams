# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict
import logging
import os
import sys
import time
import random

# Third-party
from astropy import log as logger
from emcee import EnsembleSampler
import numpy as np
import numexpr
import streamteam.potential as sp
from streamteam.inference import EmceeModel, ModelParameter, LogarithmicPrior

# Project
from .. import heliocentric_names
from ..coordinates import hel_to_gal
from .rewinder_likelihood import rewinder_likelihood
from .streamcomponent import StreamComponent

logger.setLevel(logging.DEBUG)

__all__ = ["Rewinder", "RewinderSampler"]

class Rewinder(EmceeModel):

    def __init__(self, Potential, progenitor, stars,
                 dt=-1., perfect_data=False, nsamples_init=10000, nsamples=256):
        """ Model for tidal streams that uses backwards integration to Rewind
            the positions of stars.

            Parameters
            ----------
            potential : streamteam.Potential
            progenitor : streams.Progenitor
            stars : streams.Stars
            t1,t2,dt : float
                Integration parameters.
            nsamples_init : int
                Initial number of samples to draw per star for importance sampling.
            nsamples : int
                Number of samples to use that match energy criterion.

        """

        self.parameters = OrderedDict()

        # Potential
        # for par in potential.parameters.values():
        #     self.add_parameter(par, "potential")
        # self.potential_class = potential.c_class
        for par in potential_parameters:
            self.add_parameter(par, "potential")
            self.potential_class = potential.__class__

        # Progenitor
        for par in progenitor.parameters.values():
            self.add_parameter(par, "progenitor")
        self.progenitor = progenitor

        # Stars
        for par in stars.parameters.values():
            if par.frozen is False:
                par.frozen = True
            self.add_parameter(par, "stars")

        self.stars = stars
        self.nstars = len(stars)

        self.perfect_data = perfect_data

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
            d = parameter_values['progenitor']['d']
            ln_prior += np.log((1 / np.log(200./1.))) - np.log(d)

            # this is actually the data likelihood
            ln_prior += log_normal(parameter_values['progenitor']['d'],
                                   self.progenitor.data[:,2],
                                   self.progenitor.errors[:,2])

        if parameters['progenitor']['mul'].frozen is False:
            mul = parameter_values['progenitor']['mul']
            ln_prior += log_normal(mul, 0., 0.306814 / d) # 300 km/s

            # this is actually the data likelihood
            ln_prior += log_normal(mul,
                                   self.progenitor.data[:,3],
                                   self.progenitor.errors[:,3])

        if parameters['progenitor']['mub'].frozen is False:
            mub = parameter_values['progenitor']['mub']
            ln_prior += log_normal(mub, 0., 0.306814 / d) # 300 km/s

            # this is actually the data likelihood
            ln_prior += log_normal(mub,
                                   self.progenitor.data[:,4],
                                   self.progenitor.errors[:,4])

        if parameters['progenitor']['vr'].frozen is False:
            vr = parameter_values['progenitor']['vr']
            ln_prior += log_normal(vr, 0., 0.306814) # 300 km/s

            # this is actually the data likelihood
            ln_prior += log_normal(vr,
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

        # rotation of Lagrange points in orbital plane
        if parameters['progenitor']['theta'].frozen is False:
            theta = parameter_values['progenitor']['theta']
        else:
            theta = parameters['progenitor']['theta'].frozen

        # satellite coordinates
        prog_hel = np.empty((1,6))
        for i,par_name in enumerate(heliocentric_names):
            if parameters['progenitor'][par_name].frozen is False:
                prog_hel[:,i] = parameter_values['progenitor'][par_name]
            else:
                prog_hel[:,i] = parameters['progenitor'][par_name].frozen

        # TODO: need to specify R_sun and V_circ as parameters?
        prog_gal = hel_to_gal(prog_hel)

        if self.perfect_data:
            stars_gal = hel_to_gal(self.stars.data)
            beta = np.array(self.stars.parameters['tail'].frozen)

            # Assume there is no uncertainty in the positions of the stars!
            ln_like = rewinder_likelihood(t1, t2, dt, potential,
                                          prog_gal, stars_gal,
                                          m0, mdot, alpha, beta, theta)

            likelihood = ln_like.sum()

        else:
            raise NotImplementedError()

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
        stars_obs = np.vstack([star_data[name] for name in heliocentric_names])
        try:
            stars_err = np.vstack([star_data["err_{}".format(name)] for name in heliocentric_names])
        except ValueError:
            logger.warning("Star data uncertainty columns misnamed or don't exist.")
            stars_err = None

        stars = StreamComponent(stars_obs, err=stars_err, tail=star_data['tail'])

        # -------------------------------------------------------------------------------
        prog_obs = np.vstack([prog_data[name] for name in heliocentric_names])
        try:
            prog_err = np.vstack([prog_data["err_{}".format(name)] for name in heliocentric_names])
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

        # Progenitor mass-loss
        # TODO: this is too rigid...
        try:
            mdot = float(config.get('progenitor').get('mass_loss'))
        except:
            logger.warning("Failed to get mass-loss rate from config file! Assuming no mass-loss.")
            mdot = 0.

        prog = StreamComponent(prog_obs, err=prog_err, m0=m0, mdot=mdot)

        # -------------------------------------------------------------------------------

        # Read integration stuff
        dt = float(config['integration'].get('dt'))
        nintegrate = int(config['integration'].get('nsteps'))
        logger.debug("Will integrate for {} steps, with a timestep of {} Myr"\
                     .format(nintegrate, dt))
        logger.info("Integration time: {} Myr".format(nintegrate*dt))

        # Potential
        Potential = getattr(sp, config["potential"]["class"])
        logger.info("Using potential '{}'...".format(Potential))
        # TODO: other potential stuff

        # Initialize the model
        model = cls(Potential,
                    progenitor=(prog, prog_err),
                    stars=(stars, stars_err),
                    dt=dt, nsteps=nintegrate)

        # TODO: alpha, theta

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

        super(RewinderSampler, self).__init__(self.nwalkers, model.nparameters, model,
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

