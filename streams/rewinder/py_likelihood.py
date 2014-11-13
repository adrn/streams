# coding: utf-8

""" Pure-python implementation of my likelihood function """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import numpy as np
from astropy import log as logger

def get_basis(prog_w, theta=0.):
    """ Compute the instantaneous orbital plane basis at each timestep for the
        progenitor system orbit.
    """
    basis = np.zeros((len(prog_w),3,3))

    prog_x = prog_w[...,:3]
    prog_v = prog_w[...,3:]

    x1_hat = prog_x
    x3_hat = np.cross(x1_hat, prog_v)
    x2_hat = -np.cross(x1_hat, x3_hat)

    x1_hat /= np.linalg.norm(x1_hat, axis=-1)[...,np.newaxis]
    x2_hat /= np.linalg.norm(x2_hat, axis=-1)[...,np.newaxis]
    x3_hat /= np.linalg.norm(x3_hat, axis=-1)[...,np.newaxis]

    if theta != 0.:
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        R = np.array([[costheta, sintheta,0],[-sintheta, costheta,0],[0,0,1]])

        x1_hat = x1_hat.dot(R)
        x2_hat = x2_hat.dot(R)

    basis[...,0] = x1_hat
    basis[...,1] = x2_hat
    basis[...,2] = x3_hat
    return basis

def rewinder_likelihood(dt, nsteps, potential, prog_xv, star_xv, m0, mdot,
                        alpha, betas, theta, selfgravity=False):

    # full array of initial conditions for progenitor and stars
    w0 = np.vstack((prog_xv,star_xv))

    # integrate orbits
    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps)
    t = t[:-1]
    w = w[:-1]

    # satellite mass
    sat_mass = -mdot*t + m0
    GMprog = potential.parameters['G'] * sat_mass

    # compute approximations of tidal radius and velocity dispersion from mass enclosed
    menc = potential.mass_enclosed(w[:,0,:3])  # progenitor position orbit
    E_scale = (sat_mass / menc)**(1/3.)

    # compute naive tidal radius and velocity dispersion
    rtide = E_scale * np.linalg.norm(w[:,0,:3], axis=-1)  # progenitor orbital radius
    vdisp = E_scale * np.linalg.norm(w[:,0,3:], axis=-1)  # progenitor orbital velocity

    # get the instantaneous orbital plane basis vectors (x1,x2,x3)
    basis = get_basis(w[:,0], theta)

    # star orbits relative to progenitor
    dw = w[:,1:] - w[:,0:1]

    # project orbits into new basis
    w123 = np.zeros_like(dw)
    for i in range(3):
        w123[...,i] = np.sum(dw[...,:3] * basis[...,i][:,np.newaxis], axis=-1)
        w123[...,i+3] = np.sum(dw[...,3:] * basis[...,i][:,np.newaxis], axis=-1)

    w123[...,i] += alpha*betas[np.newaxis]*rtide[:,np.newaxis]

    # write like this to allow for more general dispersions...probably want a covariance matrix
    sigmas = np.zeros((nsteps,1,6))
    sigmas[:,0,0] = rtide
    sigmas[:,0,1] = rtide
    sigmas[:,0,2] = rtide

    sigmas[:,0,3] = vdisp
    sigmas[:,0,4] = vdisp
    sigmas[:,0,5] = vdisp

    g = -0.5*np.log(2*np.pi) - np.log(sigmas) - 0.5*(w123/sigmas)**2

    # compute an estimate of the jacobian
    Rsun = 8.
    R2 = (w[:,1:,0] + Rsun)**2 + w[:,1:,1]**2 + w[:,1:,2]**2
    x2 = w[:,1:,2]**2 / R2
    log_jac = np.log(R2*R2 * np.sqrt(1.-x2))

    return g.sum(axis=-1) + log_jac, w
