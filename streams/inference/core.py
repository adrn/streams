# coding: utf-8

""" Core functionality for doing stream inference """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
from collections import OrderedDict
import time

# Third-party
from emcee import EnsembleSampler, PTSampler
import numpy as np
import astropy.units as u
from scipy.stats import truncnorm
import h5py

# Project
from .back_integrate import back_integration_likelihood
from ..coordinates.frame import heliocentric
from ..coordinates import _hel_to_gc, _gc_to_hel
from .parameter import ModelParameter
from .prior import *
from ..util import _parse_quantity
from .. import usys

__all__ = ["StreamModel", "StreamModelSampler"]

logger = logging.getLogger(__name__)

# def log_normal(x, mu, sigma, axis=-1):
#     X = x - mu
#     k = X.shape[-1]

#     term1 = -0.5*k*np.log(2*np.pi)
#     term2 = -0.5*np.sum(np.log(sigma), axis=axis)
#     term3 = -0.5*np.sum((X/sigma)**2, axis=axis)

#     return term1+term2+term3

def log_normal(x, mu, sigma):
    X = x - mu
    return -0.5*(np.log(2*np.pi) + 2*np.log(sigma) + (X/sigma)**2)

class StreamModel(object):

    def __init__(self, potential, satellite, particles,
                 true_satellite=None, true_particles=None, lnpargs=()):
        """ Model for tidal streams that uses backwards integration to Rewind
            the positions of stars.

            Parameters
            ----------
            potential : streams.potential.Potential
                The potential to fit.
            satellite : streams.dynamics.ObservedParticle
            particles : streams.dynamics.ObservedParticle
            true_satellite : streams.dynamics.Particle (optional)
            true_particles : streams.dynamics.Particle (optional)
            lnpargs : iterable (optional)
                Other arguments to the posterior function.
        """

        # store the potential class, and parameters originally specified
        self._potential_class = potential.__class__
        self._given_potential_params = potential._parameter_dict.copy()

        # internal parameter dictionary
        self.parameters = OrderedDict()
        self.parameters['potential'] = OrderedDict()
        self.parameters['particles'] = OrderedDict()
        self.parameters['satellite'] = OrderedDict()
        self.nparameters = 0

        # extra arguments passed to the posterior function
        self.lnpargs = lnpargs

        self.satellite = satellite
        self.particles = particles
        self.true_satellite = true_satellite
        self.true_particles = true_particles

        self._prior_cache = dict()

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

    def add_parameter(self, parameter_group, parameter):
        """ Add a parameter to the model in the specified parameter group.

            Parameters
            ----------
            parameter_group : str
                Name of the parameter group to add this parameter to.
            parameter : streams.inference.ModelParameter
                The parameter instance.
        """

        # TODO: need a ModelParameter object -- needs to have a .size, .shape
        if not self.parameters.has_key(parameter_group):
            # currently only supports pre-specified parameter groups
            raise KeyError("Invalid parameter group '{}'".format(parameter_group))

        if not isinstance(parameter, ModelParameter):
            raise TypeError("Invalid parameter type '{}'".format(type(parameter)))

        self.parameters[parameter_group][parameter.name] = parameter.copy()
        self.nparameters += parameter.size

    @property
    def truths(self):
        """ Returns an array of the true values of all parameters in the model """

        true_p = np.array([])
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                t = param.truth
                if t is None:
                    #t = np.ones(param.size)*np.nan
                    t = [None]*param.size
                true_p = np.append(true_p, np.ravel(t))

        return true_p

    def sample_priors(self, size=None, start_truth=False):
        """ Draw samples from the priors over the model parameters.

            Parameters
            ----------
            size : int
                Number of samples to draw.
            start_truth : bool (optional)
                Sample centered on true values.
        """

        sz = size if size is not None else 1
        p0 = np.zeros((sz, self.nparameters))
        for ii in range(sz):
            ix1 = 0
            for group_name,group in self.parameters.items():
                for param_name,param in group.items():

                    if param_name in heliocentric.coord_names:
                        err = getattr(self, group_name).errors[param_name].decompose(usys).value
                        if start_truth:
                            val = getattr(self, "true_"+group_name)[param_name]\
                                        .decompose(usys).value
                        else:
                            val = getattr(self, group_name)[param_name]\
                                        .decompose(usys).value

                        if np.any(np.isinf(err)):
                            if param_name in ["mul", "mub", "vr"]:
                                err[np.isinf(err)] = 0.1*val[np.isinf(err)]
                            else:
                                err[np.isinf(err)] = 1.

                        try:
                            prior = self._prior_cache[(group_name,param_name)]
                        except KeyError:
                            prior = LogNormalPrior(val, err)
                            self._prior_cache[(group_name,param_name)] = prior

                        p0[ii,ix1:ix1+param.size] = np.ravel(prior.sample().T)
                        ix1 += param.size

                        continue

                    if start_truth:
                        prior = param._prior
                        if hasattr(prior, 'a') and hasattr(prior, 'b'):
                            mu = np.ravel(param.truth)
                            sigma = mu/100.
                            a = (prior.a - mu) / sigma
                            b = (prior.b - mu) / sigma
                            if param.size > 1:
                                rvs = []
                                for aa,bb,mm,ss in zip(a,b,mu,sigma):
                                    X = truncnorm(aa, bb,
                                                  loc=mm, scale=ss)
                                    rvs.append(X.rvs())
                            else:
                                X = truncnorm(a, b,
                                              loc=mu, scale=sigma)
                                rvs = X.rvs()

                            p0[ii,ix1:ix1+param.size] = np.ravel(rvs)
                        elif hasattr(prior, 'mu'):
                            v = np.random.normal(param.truth, prior.sigma/2.)
                            p0[ii,ix1:ix1+param.size] = np.ravel(v)
                        else:
                            p0[ii,ix1:ix1+param.size] = np.ravel(param.truth)
                    else:
                        if param.size > 1:
                            p0[ii,ix1:ix1+param.size] = np.ravel(param.sample().T)
                        else:
                            p0[ii,ix1:ix1+param.size] = np.ravel(param.sample())

                    ix1 += param.size
        return np.squeeze(p0)

    def _decompose_vector(self, p):
        """ Turns a vector of parameter values, e.g. from MCMC, and turns it into
            a dictionary of parameters.

            Parameters
            ----------
            p : array_like
                The vector of model parameter values.
        """
        d = OrderedDict()

        ix1 = 0
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                try:
                    d[group_name][param_name] = p[ix1:ix1+param.size]
                except:
                    d[group_name] = OrderedDict()
                    d[group_name][param_name] = p[ix1:ix1+param.size]

                if d[group_name][param_name].size > 1:
                    d[group_name][param_name] = d[group_name][param_name].reshape(param.shape)
                else:
                    d[group_name][param_name] = np.squeeze(d[group_name][param_name])

                ix1 += param.size

        return d

    def _compose_vector(self, param_dict):
        """ Turn a parameter dictionary into a parameter vector

            Parameters
            ----------
            param_dict : OrderedDict
        """

        vec = np.array([])
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                try:
                    vec = np.append(vec, np.ravel(param_dict[group_name][param_name]))
                except:
                    vec = np.append(vec, np.ravel(param_dict[group_name][param_name].value))

        return vec

    def label_flatchain(self, flatchain):
        """ Turns a flattened MCMC chain (e.g., sampler.flatchain from emcee) and
            turns it into a dictionary of parameter samples.

            Parameters
            ----------
            flatchain : array_like
                The flattened MCMC chain of parameter values. Should have shape (nsteps, ndim).
        """
        d = OrderedDict()

        nsamples,ndim = flatchain.shape
        # make sure ndim == nparameters
        if ndim != self.nparameters:
            raise ValueError("Flatchain ndim != model nparameters")

        ix1 = 0
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                try:
                    d[group_name][param_name] = flatchain[:,ix1:ix1+param.size]
                except:
                    d[group_name] = OrderedDict()
                    d[group_name][param_name] = flatchain[:,ix1:ix1+param.size]

                if param.size > 1:
                    shp = (nsamples,) + param.shape
                    d[group_name][param_name] = d[group_name][param_name].reshape(shp)

                ix1 += param.size

        return d

    def ln_prior(self, param_dict, *args):
        ln_prior = 0.
        for group_name,group in self.parameters.items():
            for param_name,param in group.items():
                lp = param.prior(param_dict[group_name][param_name])
                ln_prior += lp

        return ln_prior

    def ln_likelihood(self, param_dict, *args):
        """ Evaluate the log-posterior at the given parameter set.

            Parameters
            ----------
            param_dict : dict
                The dictionary of model parameter values.
        """

        ######################################################################
        # potential parameters:
        #
        pparams = self._given_potential_params.copy()
        for k,v in param_dict.get('potential',dict()).items():
            pparams[k] = v

        ######################################################################
        # nuisance parameters:

        # particle tail assignment
        try:
            beta = param_dict['particles']['beta']
        except KeyError:
            beta = self.particles.beta.truth
            if np.any(np.isnan(beta)):
                print("True tail assignment was NaN!")
                sys.exit(1)

        # particle unbinding time
        try:
            tub = param_dict['particles']['tub']
        except KeyError:
            tub = self.particles.tub.truth

        # satellite mass
        try:
            logmass = param_dict['satellite']['logmass']
        except KeyError:
            logmass = self.satellite.logmass.truth

        # satellite mass-loss
        try:
            logmdot = param_dict['satellite']['logmdot']
        except KeyError:
            logmdot = self.satellite.logmdot.truth

        # position of effective tidal radius
        try:
            alpha = param_dict['satellite']['alpha']
        except KeyError:
            alpha = self.satellite.alpha.truth

        ######################################################################
        # Heliocentric coordinates
        #

        # first see if any coordinates are parameters. if not, assume we're
        #   running a test with no observational uncertainties
        if param_dict.has_key('particles'):
            param_names = param_dict['particles'].keys()
            num_free = 0
            for pname in param_names:
                if pname in heliocentric.coord_names:
                    num_free += 1
        else:
            num_free = 0

        # Particles
        if num_free == 0:
            # no free coordinate parameters -- use truths
            p_hel = self.true_particles._X
            data_like = 0.
        else:
            p_hel = np.zeros_like(self.particles._X)
            data_like = 0.
            for ii,k in enumerate(heliocentric.coord_names):
                if k in param_dict['particles'].keys():
                    p_hel[:,ii] = param_dict['particles'][k]
                    ll = log_normal(param_dict['particles'][k],
                                    self.particles._X[:,ii],
                                    self.particles._error_X[:,ii])
                else:
                    p_hel[:,ii] = self.particles._X[:,ii]
                    ll = 0.

                if np.any(np.isinf(ll)):
                    ll = 0.

                data_like += ll

            # if any distance is > 150 kpc
            if np.any(p_hel[...,2] > 150.):
                return -np.inf
        p_gc = _hel_to_gc(p_hel)

        # Satellite
        if param_dict.has_key('satellite'):
            param_names = param_dict['satellite'].keys()
            num_free = 0
            for pname in param_names:
                if pname in heliocentric.coord_names:
                    num_free += 1
        else:
            num_free = 0.

        if num_free == 0:
            # no free coordinate parameters -- use truths
            s_hel = self.true_satellite._X
            sat_like = 0.
        else:
            s_hel = np.zeros_like(self.satellite._X)
            sat_like = 0.
            for ii,k in enumerate(heliocentric.coord_names):
                if k in param_dict['satellite'].keys():
                    s_hel[:,ii] = param_dict['satellite'][k]
                    ll = log_normal(param_dict['satellite'][k],
                                    self.satellite._X[:,ii],
                                    self.satellite._error_X[:,ii])
                else:
                    s_hel[:,ii] = self.satellite._X[:,ii]
                    ll = 0.

                if np.any(np.isinf(ll)):
                    ll = 0.

                sat_like += ll

            # if any distance is > 150 kpc
            if np.any(s_hel[...,2] > 150.):
                return -np.inf
        s_gc = _hel_to_gc(s_hel)

        # TODO: don't create new potential each time, just modify _parameter_dict?
        potential = self._potential_class(**pparams)
        t1, t2, dt = args[:3]
        ln_like = back_integration_likelihood(t1, t2, dt,
                                              potential, p_gc, s_gc,
                                              logmass, logmdot,
                                              beta, alpha, tub)

        return np.sum(ln_like + data_like) + np.squeeze(sat_like)

    def ln_posterior(self, p, *args):

        param_dict = self._decompose_vector(p)

        ln_prior = self.ln_prior(param_dict, *args)

        # short-circuit if any prior value is -infinity
        if np.any(np.isinf(ln_prior)):
            return -np.inf

        ln_like = self.ln_likelihood(param_dict, *args)

        if np.any(np.isnan(ln_like)) or np.any(np.isnan(ln_prior)):
            return -np.inf

        try:
            return np.sum(ln_like)*args[3] + np.sum(ln_prior)
        except:
            return np.sum(ln_like) + np.sum(ln_prior)

    def __call__(self, p):
        # TODO: each call, adjust temperature according to self.annealing?
        return self.ln_posterior(p, *self.lnpargs)

class StreamModelSampler(EnsembleSampler):

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

