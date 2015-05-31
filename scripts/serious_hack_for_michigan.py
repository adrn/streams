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
    alpha, v_c = p

    if alpha < 0.2 or alpha > 3.:
        return -np.inf

    if v_c < 0.1 or v_c > 0.3: # 100 - 300 km/s
        return -np.inf

    return 0.

def ln_likelihood_nfw(p, dt, nsteps, prog_w, star_w, betas, sat_mass):
    alpha, v_c = p

    params = true_params.copy()
    params['v_c'] = v_c
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
    return -(ln_pri + ll)

# ----------------------------------------------------------------------
def ln_prior_hernq(p):
    log_sat_mass, alpha, log_m, c = p

    if log_sat_mass < 10. or log_sat_mass > 14.7: # 2.5E4 - 2.5E6
        return -np.inf

    if alpha < 0.2 or alpha > 3.:
        return -np.inf

    if log_m < 23. or log_m > 27.6:
        return -np.inf

    if c < 0.1 or c > 100.:
        return -np.inf

    return 0.

def ln_likelihood_hernq(p, dt, nsteps, prog_w, star_w, betas):
    log_sat_mass, alpha, log_m, c = p

    params = dict()
    params['m'] = np.exp(log_m)
    params['c'] = c
    pot = gp.HernquistPotential(units=galactic, **params)

    nstars = star_w.shape[0]
    ll = np.zeros((nsteps,nstars), dtype=float) - 9999.
    rewinder_likelihood(ll, dt, nsteps, pot.c_instance, pot.G, prog_w[None].copy(), star_w,
                        np.exp(log_sat_mass), 0., alpha, betas, 0., True)

    return integrate_tub(ll, dt).sum()

def ln_posterior_hernq(p, dt, nsteps, prog_w, star_w, betas):
    ln_pri = ln_prior_hernq(p)
    if np.any(~np.isfinite(ln_pri)):
        return -np.inf

    ll = ln_likelihood_hernq(p, dt, nsteps, prog_w, star_w, betas)
    if np.any(~np.isfinite(ll)):
        return -np.inf
    return -(ln_pri + ll)

# ----------------------------------------------------------------------
def ln_prior_bfe(p):
    log_sat_mass, alpha, log_m, c, c1, c2, c3, c4 = p

    if log_sat_mass < 10. or log_sat_mass > 14.7: # 2.5E4 - 2.5E6
        return -np.inf

    if alpha < 0.2 or alpha > 3.:
        return -np.inf

    if log_m < 23. or log_m > 27.6:
        return -np.inf

    if c < 0.1 or c > 100.:
        return -np.inf

    return 0.

def ln_likelihood_bfe(p, dt, nsteps, prog_w, star_w, betas):
    log_sat_mass, alpha, log_m, c, c1, c2, c3, c4 = p

    params = dict()
    params['m'] = np.exp(log_m)
    params['c'] = c
    params['sin_coeffs'] = np.array([0.]*4)
    params['cos_coeffs'] = np.array([c1, c2, c3, c4])
    pot = gp.SphericalBFEPotential(units=galactic, **params)

    nstars = star_w.shape[0]
    ll = np.zeros((nsteps,nstars), dtype=float) - 9999.
    rewinder_likelihood(ll, dt, nsteps, pot.c_instance, pot.G, prog_w[None].copy(), star_w,
                        np.exp(log_sat_mass), 0., alpha, betas, 0., True)

    return integrate_tub(ll, dt).sum()

def ln_posterior_bfe(p, dt, nsteps, prog_w, star_w, betas):
    ln_pri = ln_prior_bfe(p)
    if np.any(~np.isfinite(ln_pri)):
        return -np.inf

    ll = ln_likelihood_bfe(p, dt, nsteps, prog_w, star_w, betas)
    if np.any(~np.isfinite(ll)):
        return -np.inf
    return -(ln_pri + ll)

# =====================================================================================

def main():
    pool = get_pool(mpi=False)

    np.random.seed(42)

    # parameters
    true_sat_mass = 2.5E5
    nstars = 16
    dt = -0.1

    # read in the SCF simulation data
    s = scf.SCFReader("/Users/adrian/projects/scf/simulations/runs/spherical/")

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

    # ----------------------------
    # Emcee
    nburn = 2
    nwalk = 2

    # NFW
    print("Firing up sampler for NFW")
    ndim = 2
    nwalkers = 8*ndim
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn=ln_posterior_nfw,
                                    args=(dt, nsteps, prog_w, data_w, betas, true_sat_mass),
                                    pool=pool)

    p0 = np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.normal(1., 0.25, size=nwalkers) # alpha
    p0[:,1] = np.random.normal(0.2, 0.02, size=nwalkers) # v_c

    pos,_,_ = sampler.run_mcmc(p0, nburn)
    print("Done burning in.")
    sampler.reset()
    pos,_,_ = sampler.run_mcmc(pos, nwalk)
    print("Done sampling.")
    # np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/nfw_chain.npy", sampler.chain)
    # np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/nfw_lnprob.npy", sampler.lnprobability)

    # Hernquist
    # print("Firing up sampler for Hernquist")
    # ndim = 4
    # nwalkers = 8*ndim
    # sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn=ln_posterior_hernq,
    #                                 args=(dt, nsteps, prog_w, data_w, betas), pool=pool)

    # p0 = np.zeros((nwalkers,ndim))
    # p0[:,0] = np.random.normal(true_log_sat_mass, 0.5, size=nwalkers) # log_sat_mass
    # p0[:,1] = np.random.normal(1., 0.25, size=nwalkers) # alpha
    # p0[:,2] = np.random.normal(26., 0.15, size=nwalkers) # log_m
    # p0[:,3] = np.random.normal(20., 2., size=nwalkers) # c

    # pos,_,_ = sampler.run_mcmc(p0, nburn)
    # print("Done burning in.")
    # sampler.reset()
    # pos,_,_ = sampler.run_mcmc(pos, nwalk)
    # print("Done sampling.")
    # np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/hernq_chain.npy", sampler.chain)
    # np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/hernq_lnprob.npy", sampler.lnprobability)

    # BFE
    # print("Firing up sampler for BFE")
    # ndim = 8
    # nwalkers = 8*ndim
    # sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn=ln_posterior_bfe,
    #                                 args=(dt, nsteps, prog_w, data_w, betas), pool=pool)

    # p0 = np.zeros((nwalkers,ndim))
    # p0[:,0] = np.random.normal(true_log_sat_mass, 0.5, size=nwalkers) # log_sat_mass
    # p0[:,1] = np.random.normal(1., 0.25, size=nwalkers) # alpha
    # p0[:,2] = np.random.normal(26., 0.15, size=nwalkers) # log_m
    # p0[:,3] = np.random.normal(20., 2., size=nwalkers) # c
    # p0[:,4] = np.random.uniform(-0.9, -1.1, size=nwalkers) # c1
    # p0[:,5] = np.random.uniform(-0., -0.1, size=nwalkers) # c2
    # p0[:,6] = np.random.uniform(-0., -0.01, size=nwalkers) # c3
    # p0[:,7] = np.random.uniform(-0., -0.01, size=nwalkers) # c4

    # pos,_,_ = sampler.run_mcmc(p0, nburn)
    # print("Done burning in.")
    # sampler.reset()
    # pos,_,_ = sampler.run_mcmc(pos, nwalk)
    # print("Done sampling.")
    # #np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/bfe_chain.npy", sampler.chain)
    # #np.save("/vega/astro/users/amp2217/projects/streams/output/michigan_hack/bfe_lnprob.npy", sampler.lnprobability)

    # pool.close()
    # sys.exit(0)

if __name__ == "__main__":
    main()
