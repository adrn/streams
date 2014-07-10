# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from collections import OrderedDict
import time

# Third-party
from astropy import log as logger
from emcee import EnsembleSampler, PTSampler
import numpy as np
import astropy.units as u
import h5py

# Project
from streamteam.inference import EmceeModel
from .. import heliocentric_names
from .rewinder_likelihood import rewinder_likelihood

__all__ = ["Rewinder", "RewinderSampler"]

class Rewinder(EmceeModel):

    def __init__(self, potential, progenitor, stars, t1, t2, dt=-1.):
        """ Model for tidal streams that uses backwards integration to Rewind
            the positions of stars.

            Parameters
            ----------
            potential : Potential object
            progenitor :
            stars :
        """

        # Potential
        for par_name,par in potential.parameters:
            self.add_parameter(par, "potential")
        self.potential_class = potential.c_cls

        # Progenitor
        for par_name,par in progenitor.parameters:
            self.add_parameter(par, "progenitor")

        # Stars
        for par_name,par in stars.parameters:
            self.add_parameter(par, "stars")
        self.nstars = len(stars)

        self.args = (t1,t2,dt)

    @classmethod
    def from_config(cls, config):
        """ Construct a StreamModel from a configuration dictionary.
            Typically comes from a YAML file via `streams.io.read_config`.

            Parameters
            ----------
            config : dict
        """
        from ..io import read_hdf5

        # load satellite and particle data
        logger.debug("Reading particle/satellite data from:\n\t{}".format(config["data_file"]))
        d = read_hdf5(config["data_file"],
                      nparticles=config.get('nparticles', None),
                      particle_idx=config.get('particle_idx', None))

        true_particles = d['true_particles']
        true_satellite = d['true_satellite']
        satellite = d.get('satellite', None)
        particles = d.get('particles', None)
        nparticles = true_particles.nparticles
        logger.info("Running with {} particles.".format(nparticles))

        # integration stuff
        t1 = config.get("t1", float(d["t1"]))
        t2 = config.get("t2", float(d["t2"]))
        dt = config.get("dt", config.get('dt', -1.))
        logger.debug("Integration from {} to {}, dt={} Myr".format(t1,t2,dt))

        # get the potential object specified from the potential subpackage
        from .. import potential as sp
        Potential = getattr(sp, config["potential"]["class_name"])
        potential = Potential()
        logger.info("Using potential '{}'...".format(config["potential"]["class_name"]))

        # Initialize the empty model to add parameters to
        model = cls(potential, lnpargs=[t1,t2,dt],
                    particles=particles,
                    satellite=satellite,
                    true_satellite=true_satellite,
                    true_particles=true_particles)

        # Potential parameters
        if config["potential"]["parameters"]:
            for ii,name in enumerate(config["potential"]["parameters"]):
                p = potential.parameters[name].copy()
                logger.debug("Prior on {}: {}".format(name, p._prior))
                model.add_parameter('potential', p)

        # Particle parameters
        if config['particles']['parameters']:
            logger.debug("Particle properties added as parameters:")

            for ii,name in enumerate(config["particles"]["parameters"]):
                if name in heliocentric.coord_names:
                    priors = [LogPrior() for ii in range(nparticles)]
                    X = ModelParameter(name,
                                       value=particles[name].decompose(usys).value,
                                       prior=priors,
                                       truth=true_particles[name].decompose(usys).value)
                    model.add_parameter('particles', X)
                    logger.debug("\t\t{}".format(name))
                else:
                    p = getattr(particles,name)
                    logger.debug("Prior on {}: {}".format(name, p._prior))
                    model.add_parameter('particles', p)

            missing_dims = config["particles"].get("missing_dims", list())
            for missing_dim in missing_dims:
                ix = heliocentric.coord_names.index(missing_dim)
                model.particles._error_X[:,ix] = np.inf

        # Satellite parameters
        if config['satellite']['parameters']:
            logger.debug("Satellite properties added as parameters:")
            for ii,name in enumerate(config["satellite"]["parameters"]):

                if name in heliocentric.coord_names:
                    X = ModelParameter(name,
                                       value=satellite[name].decompose(usys).value,
                                       prior=LogPrior(),
                                       truth=true_satellite[name].decompose(usys).value)
                    model.add_parameter('satellite', X)
                    logger.debug("\t\t{}".format(name))

                elif name == 'alpha':
                    p = getattr(satellite,name)
                    logger.debug("Prior on {}: ".format(name))
                    model.add_parameter('satellite', p)

                else:
                    p = getattr(satellite,name)
                    logger.debug("Prior on {}: {}".format(name, p._prior))
                    model.add_parameter('satellite', p)

            missing_dims = config["satellite"].get("missing_dims", list())
            for missing_dim in missing_dims:
                ix = heliocentric.coord_names.index(missing_dim)
                model.satellite._error_X[:,ix] = np.inf

        return model

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
        for group_name in ['potential', 'progenitor']:
            for param in parameters[group_name].keys():
                v = value_dict[group_name][param.name]
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

        # TODO: need to figure out how to get P(D|X) values into here...
        #   - Maybe write custom ln_prior function that does all priors
        #       except the phase-space coords, then in the likelihood
        #       loop over the priors and get probs from there?

        ######################################################################
        # potential parameters:
        #
        pparams = dict()
        for par in parameters['potential']:
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
        if parameters['progenitor']['mass'].frozen is False:
            mass = parameter_values['progenitor']['mass']
        else:
            mass = parameters['progenitor']['mass'].frozen

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
        prog_helct = np.empty((1,6))
        for ii,par_name in enumerate(heliocentric_names):
            if parameters['progenitor'][par_name].frozen is False:
                prog_helct[:,i] = parameter_values['progenitor'][par_name]
            else:
                prog_helct[:,i] = parameters['progenitor'][par_name].frozen
        prog_galct = _hel_to_gc(prog_helct) # TODO: need better way to do this so can specify R_sun and V_circ if I want in the future

        ######################################################################
        # star parameters:
        #

        # tail assignment
        if parameters['stars']['beta'].frozen is False:
            beta = parameter_values['stars']['beta']
        else:
            beta = parameters['stars']['beta'].frozen

        # star coordinates
        star_helct = np.empty((self.nstars,6))
        for ii,par_name in enumerate(heliocentric_names):
            if parameters['stars'][par_name].frozen is False:
                star_helct[:,i] = parameter_values['stars'][par_name]
            else:
                star_helct[:,i] = parameters['stars'][par_name].frozen
        star_galct = _hel_to_gc(star_helct) # TODO: need better way to do this so can specify R_sun and V_circ if I want in the future

        # TODO:
        ln_like = rewinder_likelihood(t1, t2, dt, pparams,
                                      prog_galct, star_galct,
                                      logmass, logmdot, alpha, betas)

        return np.sum(ln_like + data_like) + np.squeeze(sat_like)


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

