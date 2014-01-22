#!/vega/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" Script for using the Rewinder to infer the Galactic host potential """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time
import gc
import copy
import logging
from collections import defaultdict

# Third-party
import astropy.units as u
import emcee
from emcee.utils import sample_ball
import h5py
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import scipy
scipy.seterr(all="ignore")
import triangle
import yaml

# Project
from streams import usys
from streams.coordinates.frame import heliocentric, galactocentric
from streams.dynamics import Particle, Orbit, ObservedParticle
from streams.inference import *
import streams.io as io
from streams.observation.gaia import gaia_spitzer_errors
import streams.potential as sp
from streams.util import _parse_quantity, make_path, get_pool

from streams.coordinates import _hel_to_gc, _gc_to_hel
from streams.integrate import LeapfrogIntegrator

global pool
pool = None

# Create logger
logger = logging.getLogger(__name__)

hel_units = [u.radian,u.radian,u.kpc,u.radian/u.Myr,u.radian/u.Myr,u.kpc/u.Myr]

##################################################
default_config = dict(
    save_path="/tmp/",
    nburn=1000,
    niter=10,
    nsteps_per_iter=1000,
    nsteps_final=2000,
    nparticles=32,
    nwalkers=1024,
    potential_params=["q1","qz","v_halo","phi"],
    infer_tub=True,
    infer_particles=True,
    infer_satellite=False,
    name="infer_potential",
    plot_walkers=False,
    test=True
)

def jacobian(hel):
    l,b,d,mul,mub,vr = hel.T
    cosl, sinl = np.cos(l), np.sin(l)
    cosb, sinb = np.cos(b), np.sin(b)

    gc = _hel_to_gc(hel)
    x,y,z,vx,vy,vz = gc.T

    row0 = np.zeros_like(hel.T)
    row0[0] = -d*sinl*cosb
    row0[1] = -d*cosl*sinb
    row0[2] = cosl*cosb

    row1 = np.zeros_like(hel.T)
    row1[0] = d*cosl*cosb
    row1[1] = -d*sinl*sinb
    row1[2] = sinl*cosb

    row2 = np.zeros_like(hel.T)
    row2[0] = 0.
    row2[1] = -d*cosb
    row2[2] = sinb

    row3 = [-vr*cosb*sinl + mul*d*cosb*cosl - mub*d*sinb*sinl,
            -vr*sinb*cosl - mul*d*sinb*sinl + mub*d*cosb*cosl,
            cosb*sinl*mul + sinb*cosl*mub,
            d*cosb*sinl,
            d*sinb*cosl,
            cosb*cosl]

    row4 = [vr*cosb*cosl + mul*d*cosb*sinl + mub*d*sinb*cosl,
            -vr*sinb*sinl + mul*d*sinb*cosl + mub*d*cosb*sinl,
            -cosb*cosl*mul + sinb*sinl*mub,
            -d*cosb*cosl,
            d*sinb*sinl,
            cosb*sinl]

    row5 = np.zeros_like(hel.T)
    row5[0] = 0.
    row5[1] = cosb*vr + d*sinb*mub
    row5[2] = -cosb*mub
    row5[3] = 0.
    row5[4] = -d*cosb
    row5[5] = sinb

    return np.array([row0, row1, row2, row3, row4, row5])

def ln_likelihood(t1, t2, dt, potential, p_hel, s_hel, tub):

    p_gc = _hel_to_gc(p_hel)
    s_gc = _hel_to_gc(s_hel)

    gc = np.vstack((s_gc,p_gc)).copy()
    acc = np.zeros_like(gc[:,:3])
    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    np.array(gc[:,:3]), np.array(gc[:,3:]),
                                    args=(gc.shape[0], acc))

    times, rs, vs = integrator.run(t1=t1, t2=t2, dt=dt)

    s_orbit = np.vstack((rs[:,0][:,np.newaxis].T, vs[:,0][:,np.newaxis].T)).T
    p_orbits = np.vstack((rs[:,1:].T, vs[:,1:].T)).T

    # These are the unbinding time indices for each particle
    t_idx = np.array([np.argmin(np.fabs(times - t)) for t in tub])

    ##############################################
    # p_x = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
    # s_x = np.array([s_orbit[jj,0] for jj in t_idx])

    # # r_tide = potential._tidal_radius(2.5e8, s_x)
    # r_tide = potential._tidal_radius(2.5e8, s_orbit[...,:3]) * 1.26
    # v_esc = potential._escape_velocity(2.5e8, r_tide=r_tide) / 1.4

    # sat_var = np.zeros((len(times),6))
    # sat_var[:,:3] = r_tide #potential._tidal_radius(2.5e8, s_orbit[...,:3])*1.7
    # sat_var[:,3:] += v_esc # 0.017198632325
    # cov = sat_var**2

    # Sigma = np.array([cov[jj] for jj in t_idx])

    # log_p_x_given_phi = -0.5*np.sum(np.log(Sigma) + (p_x-s_x)**2/Sigma, axis=1)

    # p_x_hel = _gc_to_hel(p_x)
    # J = jacobian(p_x_hel).T
    # jac = np.array([np.linalg.slogdet(np.linalg.inv(jj))[1] for jj in J])

    # return np.sum(log_p_x_given_phi + jac)
    ##############################################
    # New Gaussian shell idea:
    p_x = np.array([p_orbits[jj,ii] for ii,jj in enumerate(t_idx)])
    s_x = np.array([s_orbit[jj,0] for jj in t_idx])

    r_tide = potential._tidal_radius(2.5e8, s_x)
    v_esc = potential._escape_velocity(2.5e8, r_tide=r_tide)
    v_disp = 0.017198632325

    rel_x = p_x-s_x
    R = np.sqrt(np.sum(rel_x[...,:3]**2, axis=-1))
    V = np.sqrt(np.sum(rel_x[...,3:]**2, axis=-1))
    lnR = np.log(R)
    lnV = np.log(V)

    #sigma_r = np.zeros_like(r_tide) + r_tide.mean()
    #mu_r = np.log(r_tide) # 1.5
    #sigma_r = np.zeros_like(r_tide) + 0.45
    v = 1.
    sigma_r = np.sqrt(np.log(1 + v/r_tide**2))
    mu_r = np.log(r_tide**2 / np.sqrt(v + r_tide**2))
    r_term = -0.5*(2*np.log(sigma_r) + ((lnR-mu_r)/sigma_r)**2) - np.log(R**3)

    # print(mu_r, sigma_r)
    # print(r_term)

    #sigma_v = np.zeros_like(v_esc) + v_esc.mean() / 1.4
    # mu_v = np.log(v_esc) #-4.1
    # sigma_v = np.zeros_like(v_esc) + 0.5
    v = v_disp
    sigma_v = np.sqrt(np.log(1 + v/v_esc**2))
    mu_v = np.log(v_esc**2 / np.sqrt(v + v_esc**2))
    v_term = -0.5*(2*np.log(sigma_v) + ((lnV-mu_v)/sigma_v)**2) - np.log(V**3)

    # print(mu_v, sigma_v)
    # print(v_term)
    #sys.exit(0)

    return np.sum(r_term + v_term)

def ln_data_likelihood(x, D, sigma):
    log_p_D_given_x = -0.5*np.sum(2.*np.log(sigma) + (x-D)**2/sigma**2, axis=-1)
    return np.sum(log_p_D_given_x)

def ln_posterior(p, *args):
    (t1, t2, dt, priors,
     s_hel, p_hel, s_hel_errors, p_hel_errors,
     potential_params, tub, nparticles) = args

    ll = 0. # ln likelihood
    lp = 0. # ln prior
    prior_ix = 0

    Npp = len(potential_params)

    ix1 = 0
    ix2 = Npp
    kwargs = dict(zip(potential_params, p[ix1:ix2]))
    potential = sp.LawMajewski2010(**kwargs)
    ix1 = ix2
    for ii in range(Npp):
        lp += priors[prior_ix](p[ii])
        prior_ix += 1

    if tub is None:
        ix2 = ix1 + nparticles
        tub = p[ix1:ix2]
        ix1 = ix2
        lp += priors[prior_ix](tub)
        prior_ix += 1

    if p_hel_errors is not None:
        ix2 = ix1 + nparticles*6
        walker_p_hel = p[ix1:ix2].reshape(nparticles,6)
        ix1 = ix2

        ll += ln_data_likelihood(walker_p_hel, p_hel, p_hel_errors)
    else:
        walker_p_hel = p_hel

    if s_hel_errors is not None:
        ix2 = ix1 + 6
        walker_s_hel = p[ix1:ix2]
        ix1 = ix2

        ll += ln_data_likelihood(walker_s_hel, s_hel, s_hel_errors)
    else:
        walker_s_hel = s_hel

    lp = np.sum(lp)
    if np.isinf(lp):
        return lp

    else:
        ll += ln_likelihood(t1, t2, dt, potential, walker_p_hel, walker_s_hel, tub)
        return lp + ll

def convert_units(X, u1, u2):
    """ Convert X from units u1 to units u2. """

    new_X = np.zeros_like(X)
    for ii in range(len(X.T)):
        new_X[...,ii] = (X[...,ii]*u1[ii]).to(u2[ii]).value
    return new_X

def main(c, mpi=False, threads=None, overwrite=False):
    """ TODO: """

    pool = get_pool(mpi=mpi, threads=threads)

    # Contains observed and true data
    data_file = "N128.hdf5"
    path = os.path.join(c["save_path"], "output_data", c["name"])
    try:
        d_path = os.path.join(os.environ["STREAMSPATH"],
                              "data/observed_particles/")
    except KeyError:
        raise ValueError("Env var $STREAMSPATH not set!")
        sys.exit(1)

    print(os.path.join(d_path, data_file))
    d = io.read_hdf5(os.path.join(d_path, data_file))
    print(d.keys())
    output_file = os.path.join(path, "inference.hdf5")

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(output_file) and overwrite:
        logger.info("Writing over output file '{}'".format(output_file))
        os.remove(output_file)

    truths = []
    potential = sp.LawMajewski2010()
    Npp = len(c["potential_params"])

    if c["infer_satellite"]:
        observed_satellite_hel = d["satellite"]._X
        satellite_hel_err = d["satellite"]._error_X
    else:
        observed_satellite_hel = d["true_satellite"]._X
        satellite_hel_err = None
    logger.debug("Read in satellite".format(observed_satellite_hel))

    if c["infer_particles"]:
        particles_hel_err = d["particles"]._error_X[:c["nparticles"]]
        observed_particles_hel = d["particles"]._X[:c["nparticles"]]
    else:
        observed_particles_hel = d["true_particles"]._X[:c["nparticles"]]
        particles_hel_err = None
    assert len(np.ravel(observed_particles_hel))==len(np.unique(np.ravel(observed_particles_hel)))

    try:
        true_tub = d["true_particles"].tub[:c["nparticles"]]
    except KeyError:
        true_tub = d["particles"].tub[:c["nparticles"]]

    if c["infer_tub"]:
        tub = None
    else:
        tub = np.array(true_tub)
    c["nparticles"] = observed_particles_hel.shape[0]
    logger.debug("Read in {} particles".format(c["nparticles"]))

    t1 = float(d["t1"])
    t2 = float(d["t2"])
    dt = -1.

    logger.debug("{} walkers".format(c["nwalkers"]))

    priors = []
    # potential
    for p_name in c["potential_params"]:
        pp = potential.parameters[p_name]
        priors.append(LogUniformPrior(*pp._range))
        truths.append(pp._truth)

    # tub
    if tub is None:
        priors.append(LogUniformPrior([t2]*c["nparticles"], [t1]*c["nparticles"]))
        truths = truths + true_tub.tolist()

    # particles
    if c["infer_particles"]:
        assert particles_hel_err is not None

        for ii in range(c["nparticles"]):
            prior = LogNormalPrior(observed_particles_hel[ii], sigma=particles_hel_err[ii])
            priors.append(prior)

        if d.has_key("true_particles"):
            truths = truths + np.ravel(d["true_particles"]._X).tolist()

    # satellite
    # TODO

    # ----------------------------------------------------------------------
    # TEST
    #
    if c["test"]:
        true_particles_hel = d["true_particles"]._X[:c["nparticles"]]

        print("true", ln_likelihood(t1, t2, dt, potential,
                      true_particles_hel, observed_satellite_hel, true_tub))
        wrong_potential = sp.LawMajewski2010(qz=1.7)
        print("wrong", ln_likelihood(t1, t2, dt, wrong_potential,
                       true_particles_hel, observed_satellite_hel, true_tub))
        #sys.exit(0)

        ######
        # check that posterior is sum of likelihoods
        np.random.seed(42)
        p = np.random.normal(observed_particles_hel, particles_hel_err)
        l1 = ln_likelihood(t1, t2, dt, potential,
                           p, observed_satellite_hel, true_tub)
        l2 = ln_data_likelihood(x=p, D=observed_particles_hel, sigma=particles_hel_err)

        args = (t1, t2, dt, priors,
                observed_satellite_hel, observed_particles_hel,
                satellite_hel_err, particles_hel_err,
                [], true_tub, c["nparticles"])

        lp = ln_posterior(np.ravel(p), *args)
        print(l1,l2,l1+l2)
        print(lp)
        ######

        true_p = [potential.parameters[p_name]._truth for p_name in c["potential_params"]]
        true_p = np.append(true_p, true_tub)
        true_p = np.append(true_p, np.ravel(true_particles_hel))

        args = (t1, t2, dt, priors,
                observed_satellite_hel, observed_particles_hel,
                satellite_hel_err, particles_hel_err,
                c["potential_params"], None, c["nparticles"])

        # # test potential
        vals = np.linspace(0.95, 1.05, 50)
        for ii,p_name in enumerate(c["potential_params"]):
            pp = potential.parameters[p_name]
            Ls = []
            for val in vals:
                #potential = sp.LawMajewski2010(**{p_name:p._truth*val})
                # ll = ln_likelihood(t1, t2, dt, potential,
                #                    true_particles_hel, observed_satellite_hel,
                #                    true_tub)
                p = true_p.copy()
                p[ii] = pp._truth*val
                ll = ln_posterior(p, *args)
                Ls.append(ll)

            plt.clf()
            plt.plot(vals*pp._truth, np.exp(Ls-max(Ls)))
            plt.axvline(pp._truth)
            plt.ylabel("Posterior")
            plt.savefig(os.path.join(path, "potential_{}.png".format(p_name)))

            plt.clf()
            plt.plot(vals*pp._truth, Ls)
            plt.axvline(pp._truth)
            plt.ylabel("Log Posterior")
            plt.savefig(os.path.join(path, "log_potential_{}.png".format(p_name)))

        sys.exit(0)

        # test tub
        tubs = np.linspace(0.,6200,25)
        for ii in range(c["nparticles"]):
            Ls = []
            Ps = []
            for tub in tubs:
                this_tub = true_tub.copy()
                this_tub[ii] = tub
                ll = ln_likelihood(t1, t2, dt, potential,
                                   observed_particles_hel, observed_satellite_hel,
                                   this_tub)
                Ls.append(ll)

                p = true_p.copy()
                p[Npp+ii] = tub
                lp = ln_posterior(p, *args)
                Ps.append(lp)

            plt.clf()
            plt.subplot(211)
            plt.plot(tubs, np.exp(Ls))
            plt.axvline(true_tub[ii])
            plt.ylabel("Likelihood")

            plt.subplot(212)
            plt.plot(tubs, np.exp(Ps))
            plt.axvline(true_tub[ii])
            plt.ylabel("Posterior")
            plt.savefig(os.path.join(path, "particle{}_tub.png".format(ii)))


            plt.clf()
            plt.subplot(211)
            plt.plot(tubs, Ls)
            plt.axvline(true_tub[ii])
            plt.ylabel("Likelihood")

            plt.subplot(212)
            plt.plot(tubs, Ps)
            plt.axvline(true_tub[ii])
            plt.ylabel("Posterior")
            plt.savefig(os.path.join(path, "log_particle{}_tub.png".format(ii)))

        #sys.exit(0)

        # particle positions
        coords = ["l","b","D","mul","mub","vr"]
        for ii in range(c["nparticles"]):
            for jj in range(6):
                vals = np.linspace(observed_particles_hel[ii,jj] - 3*particles_hel_err[ii,jj],
                                   observed_particles_hel[ii,jj] + 3*particles_hel_err[ii,jj],
                                   25)
                p_hel = true_particles_hel.copy()
                Ls = []
                Ls2 = []
                Ps = []
                for v in vals:
                    p = true_particles_hel.copy()
                    p[ii,jj] = v
                    ll = ln_data_likelihood(x=p,
                                            D=observed_particles_hel,
                                            sigma=particles_hel_err)

                    ll2 = ln_likelihood(t1, t2, dt, potential,
                                        p, observed_satellite_hel,
                                        true_tub)

                    p = true_p.copy()
                    p[Npp+c["nparticles"]+ii*6+jj] = v
                    lp = ln_posterior(p, *args)

                    Ls.append(ll)
                    Ls2.append(ll2)
                    Ps.append(lp)

                plt.clf()
                fig,axes = plt.subplots(4,1,sharex=True,figsize=(8,12))
                axes[0].plot(vals, np.exp(Ls))
                axes[0].set_ylabel("Likelihood 1")

                axes[1].plot(vals, np.exp(Ls2))
                axes[1].set_ylabel("Data likelihood")

                axes[2].plot(vals, np.exp(np.array(Ls) + np.array(Ls2)))
                axes[2].set_ylabel("Likelihood sum")

                axes[3].plot(vals, np.exp(Ps))
                axes[3].set_ylabel("Posterior")

                for ax in axes:
                    ax.axvline(true_particles_hel[ii,jj], label='true')
                    ax.axvline(observed_particles_hel[ii,jj], color='r', label='observed')
                axes[0].legend()

                fig.savefig(os.path.join(path, "particle{}_coord{}.png".format(ii,coords[jj])))

                plt.clf()
                fig,axes = plt.subplots(4,1,sharex=True,figsize=(8,12))
                axes[0].plot(vals, Ls)
                axes[0].set_ylabel("Likelihood 1")

                axes[1].plot(vals, Ls2)
                axes[1].set_ylabel("Data likelihood")

                axes[2].plot(vals, np.array(Ls)+np.array(Ls2))
                axes[2].set_ylabel("Likelihood sum")

                axes[3].plot(vals, Ps)
                axes[3].set_ylabel("Posterior")

                for ax in axes:
                    ax.axvline(true_particles_hel[ii,jj], label='true')
                    ax.axvline(observed_particles_hel[ii,jj], color='r', label='observed')
                axes[0].legend()

                fig.savefig(os.path.join(path,"log_particle{}_coord{}.png".format(ii,coords[jj])))
                plt.close('all')
            break

        return
    # ----------------------------------------------------------------------

    args = (t1, t2, dt, priors,
            observed_satellite_hel, observed_particles_hel,
            satellite_hel_err, particles_hel_err,
            c["potential_params"], tub, c["nparticles"])

    if not os.path.exists(output_file):
        logger.info("Output file '{}' doesn't exist".format(output_file))
        # get initial walker positions
        for jj in range(c["nwalkers"]):
            try:
                p0[jj] = np.concatenate([np.atleast_1d(p.sample()) for p in priors])
            except NameError:
                this_p0 = np.concatenate([np.atleast_1d(p.sample()) for p in priors])
                p0 = np.zeros((c["nwalkers"],len(this_p0)))
                p0[jj] = this_p0

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
        ### HACK TO INITIALIZE WALKERS NEAR true tub!
        for ii in range(c["nparticles"]):
            jj = ii + len(c["potential_params"])
            p0[:,jj] = np.random.normal(true_tub[ii], 1000., size=c["nwalkers"])

        # jj = c["nparticles"] + len(c["potential_params"])
        # for ii in range(c["nwalkers"]):
        #     p0[ii,jj:] = np.random.normal(np.ravel(d["true_particles"]._X[:c["nparticles"]]),
        #                                   np.ravel(particles_hel_err)*0.01)
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

        logger.debug("p0 shape: {}".format(p0.shape))
        sampler = emcee.EnsembleSampler(c["nwalkers"], p0.shape[-1], ln_posterior,
                                        pool=pool, args=args)

        total = c["nburn"] + c["nsteps_final"] + c["niter"]*c["nsteps_per_iter"]
        logger.info("Running sampler for {} steps total".format(total))

        if c["nburn"] > 0:
            logger.info("Burning in sampler for {} steps...".format(c["nburn"]))
            pos, xx, yy = sampler.run_mcmc(p0, c["nburn"])
            sampler.reset()
        else:
            pos = p0

        logger.info("Running sampler for {} iterations of {} steps..."\
                    .format(c["niter"], c["nsteps_per_iter"]))

        a = time.time()
        for ii in range(c["niter"]):
            pos, prob, state = sampler.run_mcmc(pos, c["nsteps_per_iter"])
            best_pos = sampler.flatchain[sampler.flatlnprobability.argmax()]
            std = np.std(sampler.flatchain, axis=0) / 2.
            print(best_pos[:5], std[:5]*2.)

            pos = sample_ball(best_pos, std, size=c["nwalkers"])

        pos, prob, state = sampler.run_mcmc(pos, c["nsteps_final"])

        t = time.time() - a
        logger.debug("Spent {} seconds on sampler...".format(t))

        with h5py.File(output_file, "w") as f:
            f["chain"] = sampler.chain
            f["flatchain"] = sampler.flatchain
            f["lnprobability"] = sampler.lnprobability
            f["p0"] = p0
            f["acceptance_fraction"] = sampler.acceptance_fraction
    else:
        logger.info("Output file '{}' already exists - reading in data".format(output_file))

    with h5py.File(output_file, "r") as f:
        chain = f["chain"].value
        flatchain = f["flatchain"].value
        p0 = f["p0"].value
        acceptance_fraction = f["acceptance_fraction"].value

    logger.debug("Acceptance fractions: {}".format(acceptance_fraction))

    if pool is not None:
        pool.close()

    # Plotting!
    # =========
    if c["plot_walkers"]:
        logger.info("Plotting individual walkers...")
        for jj in range(p0.shape[-1]):
            logger.debug("\t plotting walker {}".format(jj))
            plt.clf()
            for ii in range(c["nwalkers"]):
                plt.plot(chain[ii,:,jj], alpha=0.4, drawstyle='steps')

            plt.axhline(truths[jj], color='k', lw=4., linestyle='--')
            plt.savefig(os.path.join(path, "walker_{}.png".format(jj)))

    # Make corner plots
    # -----------------

    # potential
    ix1 = 0
    ix2 = Npp

    if Npp > 0:
        logger.info("Plotting potential posterior...")
        d_units = [u.s,u.km,u.deg]
        potential = sp.LawMajewski2010()
        corner_kwargs = defaultdict(list)
        corner_kwargs["xs"] = np.zeros_like(flatchain[:,ix1:ix2])
        for ii, p_name in enumerate(c["potential_params"]):
            pp = potential.parameters[p_name]
            corner_kwargs["xs"][:,ix1+ii] = (flatchain[:,ix1+ii]*pp._unit).decompose(d_units).value
            corner_kwargs["truths"].append(pp.truth.decompose(d_units).value)
            corner_kwargs["extents"].append([x.decompose(d_units).value for x in pp.range])
            if pp._unit is not u.dimensionless_unscaled:
                label = "{0} [{1}]".format(pp.latex, pp.truth.decompose(d_units).unit)
            else:
                label = "{0}".format(pp.latex)
            corner_kwargs["labels"].append(label)

        fig = triangle.corner(**corner_kwargs)
        fig.savefig(os.path.join(path, "potential_posterior.png"))
        plt.close('all')
        ix1 = ix2

    chain_nwalkers,chain_nsteps,op = chain.shape
    tub_chain = None
    if c["infer_tub"]:
        ix2 = ix1 + c["nparticles"]
        tub_chain = flatchain[:,ix1:ix2].reshape(chain_nwalkers*chain_nsteps,c["nparticles"],1)
        ix1 = ix2

    if c["infer_particles"]:
        logger.info("Plotting particle posteriors...")
        ix2 = ix1 + c["nparticles"]*6
        particles_flatchain = flatchain[:,ix1:ix2].reshape(chain_nsteps*chain_nwalkers,c["nparticles"],6)
        particles_p0 = p0[:,ix1:ix2].reshape(chain_nwalkers,c["nparticles"],6)

        for ii in range(c["nparticles"]):
            logger.debug("\tplotting particle {}".format(ii))
            this_p0 = convert_units(particles_p0[:,ii], hel_units, heliocentric.repr_units)
            true_X = convert_units(d["true_particles"]._X[ii], hel_units, heliocentric.repr_units)
            obs_X = convert_units(d["particles"]._X[ii], hel_units, heliocentric.repr_units)
            chain_X = convert_units(particles_flatchain[:,ii], hel_units, heliocentric.repr_units)

            corner_kwargs = defaultdict(list)
            corner_kwargs["xs"] = chain_X
            corner_kwargs["truths"] = true_X.tolist()
            mu,sigma = np.mean(this_p0, axis=0),np.std(this_p0, axis=0)
            corner_kwargs["extents"] = zip(mu-3*sigma,mu+3*sigma)
            labels = ["{0} [{1}]".format(n,uu)
                      for n,uu in zip(heliocentric.coord_names,heliocentric.repr_units)]
            corner_kwargs["labels"] = labels

            if tub_chain is not None:
                corner_kwargs["xs"] = np.hstack((corner_kwargs["xs"],tub_chain[:,ii]))
                corner_kwargs["truths"].append(true_tub[ii])
                corner_kwargs["extents"].append((t2,t1))
                corner_kwargs["labels"].append("$t_{ub}$")

            fig = triangle.corner(**corner_kwargs)
            fig.savefig(os.path.join(path, "particle{}_posterior.png".format(ii)))

            corner_kwargs["truths"][:-1] = obs_X.tolist()
            fig = triangle.corner(**corner_kwargs)
            fig.savefig(os.path.join(path, "particle{}_posterior_obs.png".format(ii)))

            plt.close('all')
            gc.collect()
        ix1 = ix2

    # TODO: satellite
    return
    if s_hel is None:
        ix2 = ix1 + 6
        s_hel = p[ix1:ix2]
        ix1 = ix2

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose", default=False,
                        help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                    default=False, help="Be quiet! (default = False)")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False, action="store_true",
                        help="Overwrite any existing data.")
    parser.add_argument("--test", dest="test", default=False, action="store_true",
                        help="Run tests then exit.")

    # threading
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--threads", dest="threads", default=None, type=int,
                        help="Number of multiprocessing threads to run on.")
    parser.add_argument("--machine", dest="machine", type=str, required=True,
                        choices=["yeti", "hotfoot", "deimos", "laptop"],
                        help="What machine you're running on. e.g., 'yeti' or 'hotfoot'")
    parser.add_argument("--save-path", dest="save_path", default=None, type=str,
                        help="Overwrites path set by machine name.")
    parser.add_argument("--name", dest="name", default=default_config["name"], \
                        type=str, help=".")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    config = default_config.copy()
    if args.machine.lower().strip() == "yeti":
        config["save_path"] = "/vega/astro/users/amp2217/"
    elif args.machine.lower().strip() == "hotfoot":
        config["save_path"] = "/hpc/astro/users/amp2217/"
    elif args.machine.lower().strip() == "deimos":
        config["save_path"] = "/home/adrian/projects/streams/plots/output_data"
    elif args.machine.lower().strip() == "laptop":
        config["save_path"] = "/Users/adrian/"
    else:
        raise ValueError("Unknown machine '{}'".format(args.machine))

    # overwrite the default save path for a particular machine
    if args.save_path is not None:
        config["save_path"] = args.save_path

    config["test"] = args.test
    config["name"] = args.name

    try:
        main(config, mpi=args.mpi, threads=args.threads,
             overwrite=args.overwrite)
    except:
        raise
        sys.exit(1)

    sys.exit(0)
