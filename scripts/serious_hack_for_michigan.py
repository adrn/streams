# coding: utf-8
from __future__ import division, print_function

# Std lib
import sys
import os

# hack to add project to path so we can import it without installing
biffpath = os.path.basename(os.path.abspath(__file__))
if biffpath not in sys.path:
    sys.path.append(biffpath)

scfpath = os.path.abspath(os.path.join(biffpath, "../../scf/"))
if scfpath not in sys.path:
    sys.path.append(scfpath)

# Third-party
import astropy.units as u
import gary.dynamics as gd
import gary.potential as gp
from gary.units import galactic
from gary.util import get_pool
import matplotlib.pyplot as plt
import numpy as np
import scf
import emcee
import scipy.optimize as so

from streams.rewinder import integrate_tub
from streams.rewinder.likelihood import rewinder_likelihood

true_params = dict(v_c=(200*u.km/u.s).decompose(galactic).value, r_s=20.,
                   a=1., b=1., c=1.)
true_potential = gp.LeeSutoTriaxialNFWPotential(units=galactic, **true_params)

def ln_prior_nfw(p):
    alpha, v_c, log_r_s = p

    lnp = 0.

    if alpha < 0.2 or alpha > 3.:
        return -np.inf
    lnp += -((alpha - 1.) / 0.5)**2

    if log_r_s < 0.7 or log_r_s > 4.5:
        return -np.inf

    if v_c < 0.1 or v_c > 0.3: # 100 - 300 km/s
        return -np.inf

    return lnp

def ln_likelihood_nfw(p, dt, nsteps, prog_w, star_w, betas, sat_mass):
    alpha, v_c, log_r_s = p

    params = true_params.copy()
    params['v_c'] = v_c
    params['r_s'] = np.exp(log_r_s)
    pot = gp.LeeSutoTriaxialNFWPotential(units=galactic, **params)

    nstars = star_w.shape[0]
    ll = np.zeros((nsteps,nstars), dtype=float) - 9999.
    rewinder_likelihood(ll, dt, nsteps, pot.c_instance, pot.G, prog_w[None].copy(), star_w,
                        sat_mass, 0., alpha, betas, 0., True)

    return integrate_tub(ll, dt).sum()

def ln_posterior_nfw(p, *args):
    ln_pri = ln_prior_nfw(p)
    if np.any(~np.isfinite(ln_pri)):
        return -np.inf

    ll = ln_likelihood_nfw(p, *args)
    if np.any(~np.isfinite(ll)):
        return -np.inf
    return (ln_pri + ll)

# ----------------------------------------------------------------------
def ln_prior_hernq(p):
    alpha, log_m, c = p

    lnp = 0.

    if alpha < 0.2 or alpha > 3.:
        return -np.inf
    lnp += -((alpha - 1.) / 0.5)**2

    if log_m < 23. or log_m > 27.6:
        return -np.inf

    if c < 0.1 or c > 100.:
        return -np.inf

    return lnp

def ln_likelihood_hernq(p, dt, nsteps, prog_w, star_w, betas, sat_mass):
    alpha, log_m, c = p

    params = dict()
    params['m'] = np.exp(log_m)
    params['c'] = c
    pot = gp.HernquistPotential(units=galactic, **params)

    nstars = star_w.shape[0]
    ll = np.zeros((nsteps,nstars), dtype=float) - 9999.
    rewinder_likelihood(ll, dt, nsteps, pot.c_instance, pot.G, prog_w[None].copy(), star_w,
                        sat_mass, 0., alpha, betas, 0., True)

    return integrate_tub(ll, dt).sum()

def ln_posterior_hernq(p, *args):
    ln_pri = ln_prior_hernq(p)
    if np.any(~np.isfinite(ln_pri)):
        return -np.inf

    ll = ln_likelihood_hernq(p, *args)
    if np.any(~np.isfinite(ll)):
        return -np.inf
    return (ln_pri + ll)

# ----------------------------------------------------------------------
def ln_prior_bfe(p):
    alpha = p[0]
    log_m = p[1]
    c = p[2]

    lnp = 0.

    if alpha < 0.2 or alpha > 3.:
        return -np.inf
    lnp += -((alpha - 1.) / 0.5)**2

    if log_m < 23. or log_m > 27.6:
        return -np.inf

    if c < 0.1 or c > 100.:
        return -np.inf

    for cn in p[3:]:
        if cn < 0.:
            return -np.inf

    return lnp

def ln_likelihood_bfe(p, dt, nsteps, prog_w, star_w, betas, sat_mass):
    alpha = p[0]
    log_m = p[1]
    c = p[2]

    params = dict()
    params['m'] = np.exp(log_m)
    params['c'] = c
    params['coeffs'] = np.array(p[3:])
    pot = gp.SphericalBFEPotential(units=galactic, **params)

    nstars = star_w.shape[0]
    ll = np.zeros((nsteps,nstars), dtype=float) - 9999.
    rewinder_likelihood(ll, dt, nsteps, pot.c_instance, pot.G, prog_w[None].copy(), star_w,
                        sat_mass, 0., alpha, betas, 0., True)

    return integrate_tub(ll, dt).sum()

def ln_posterior_bfe(p, *args):
    ln_pri = ln_prior_bfe(p)
    if np.any(~np.isfinite(ln_pri)):
        return -np.inf

    ll = ln_likelihood_bfe(p, *args)
    if np.any(~np.isfinite(ll)):
        return -np.inf
    return (ln_pri + ll)

# =====================================================================================

def sample_dat_ish(sampler, p0, nburn=128, nwalk=1024):
    pos,_,_ = sampler.run_mcmc(p0, nburn)
    print("Done burning in.")

    # find MAP sample, re-initialize a small ball around here
    pvec = sampler.flatchain[sampler.flatlnprobability.argmax()]
    pos = np.random.normal(pvec, np.abs(pvec)*0.01, size=p0.shape)

    sampler.reset()
    pos,_,_ = sampler.run_mcmc(pos, nburn)
    print("Second stage burn complete.")

    sampler.reset()
    pos,_,_ = sampler.run_mcmc(pos, nwalk)
    print("Done final walking.")

    return sampler

def main(name, nstars, nburn, nwalk, mpi=False, save=False):
    pool = get_pool(mpi=mpi)

    np.random.seed(42)

    # parameters
    true_sat_mass = 2.5E5
    nstars = 8
    dt = -0.1

    # read in the SCF simulation data

    s = scf.SCFReader(os.path.join(scfpath, "simulations/runs/spherical/"))

    tbl = s.last_snap(units=galactic)
    total_time = tbl.meta['time']
    nsteps = abs(int(total_time/dt))

    stream_tbl = tbl[(tbl["tub"] != 0)]
    prog_w = np.median(scf.tbl_to_w(tbl[(tbl["tub"] == 0)]), axis=0)

    # pluck out a certain number of stars...
    ixs = []
    while len(ixs) < nstars:
        ixs = np.unique(np.random.randint(len(stream_tbl), size=nstars))
    data_w = scf.tbl_to_w(stream_tbl[ixs])

    prog_E = true_potential.total_energy(prog_w[:3], prog_w[3:])
    dE = (true_potential.total_energy(data_w[:,:3], data_w[:,3:]) - prog_E) / prog_E
    betas = -2*(dE > 0).astype(int) + 1.

    # test true potential
    # vals = np.linspace(0.1, 3.5, 32)
    # lls = []
    # for val in vals:
    #     p = [val, true_params['v_c'], np.log(20.)]
    #     ll = ln_posterior_nfw(p, dt, nsteps, prog_w, data_w, betas, true_sat_mass)
    #     lls.append(ll)

    # plt.figure()
    # plt.plot(vals, lls)
    # plt.show()

    # return

    # ----------------------------
    # Emcee

    if name == 'nfw':
        # NFW
        print("Firing up sampler for NFW")
        ndim = 3
        nwalkers = 8*ndim
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn=ln_posterior_nfw,
                                        args=(dt, nsteps, prog_w, data_w, betas, true_sat_mass),
                                        pool=pool)

        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.normal(1., 0.05, size=nwalkers) # alpha
        p0[:,1] = np.random.normal(0.2, 0.02, size=nwalkers) # v_c
        p0[:,2] = np.random.normal(3., 0.1, size=nwalkers) # log_r_s

        sampler = sample_dat_ish(sampler, p0, nburn=nburn, nwalk=nwalk)
        pool.close()

        if save:
            np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/nfw_chain.npy", sampler.chain)
            np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/nfw_lnprob.npy", sampler.lnprobability)

    elif name == "hernq":
        # Hernquist
        print("Firing up sampler for Hernquist")
        ndim = 3
        nwalkers = 8*ndim
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn=ln_posterior_hernq,
                                        args=(dt, nsteps, prog_w, data_w, betas, true_sat_mass),
                                        pool=pool)

        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.normal(1., 0.05, size=nwalkers) # alpha
        p0[:,1] = np.random.normal(26., 0.15, size=nwalkers) # log_m
        p0[:,2] = np.random.normal(20., 0.5, size=nwalkers) # c

        sampler = sample_dat_ish(sampler, p0, nburn=nburn, nwalk=nwalk)
        pool.close()

        if save:
            np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/hernq_chain.npy", sampler.chain)
            np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/hernq_lnprob.npy", sampler.lnprobability)

    elif name == "bfe":
        # BFE
        print("Firing up sampler for BFE")
        ndim = 7
        nwalkers = 8*ndim
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn=ln_posterior_bfe,
                                        args=(dt, nsteps, prog_w, data_w, betas, true_sat_mass),
                                        pool=pool)

        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.normal(1., 0.05, size=nwalkers) # alpha
        p0[:,1] = np.random.normal(26., 0.15, size=nwalkers) # log_m
        p0[:,2] = np.random.normal(20., 0.5, size=nwalkers) # c
        p0[:,3] = np.random.uniform(0.9, 1.1, size=nwalkers) # c1
        p0[:,4] = np.random.uniform(0., 0.1, size=nwalkers) # c2
        p0[:,5] = np.random.uniform(0., 0.05, size=nwalkers) # c3
        p0[:,6] = np.random.uniform(0., 0.02, size=nwalkers) # c4
        # p0[:,7] = np.random.uniform(-0., -0.02, size=nwalkers) # c5
        # p0[:,8] = np.random.uniform(-0., -0.02, size=nwalkers) # c6

        sampler = sample_dat_ish(sampler, p0, nburn=nburn, nwalk=nwalk)
        pool.close()

        if save:
            np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/bfe_chain.npy", sampler.chain)
            np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/bfe_lnprob.npy", sampler.lnprobability)
    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    # name, nstars, nburn, nwalk, mpi=False, save=False
    parser.add_argument("--name", dest="name", required=True,
                        type=str, help="Potential code name (nfw, bfe, hernq).")
    parser.add_argument("--nstars", dest="nstars", required=True,
                        type=int, help="Number of stars to use.")
    parser.add_argument("--nburn", dest="nburn", default=128,
                        type=int, help="Number of steps to burn in.")
    parser.add_argument("--nwalk", dest="nwalk", default=1024,
                        type=int, help="Number of steps to walk post-burn-in.")
    parser.add_argument("--mpi", dest="mpi", action="store_true", default=False,
                        help="Use MPI.")
    parser.add_argument("--save", dest="save", action="store_true", default=False,
                        help="Save chains.")

    args = parser.parse_args()

    main(name=args.name, nstars=args.nstars, nburn=args.nburn, nwalk=args.nwalk,
         save=args.save, mpi=args.mpi)
