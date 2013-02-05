# coding: utf-8

""" This script contains some examples of how to do some simple integration in
    a specified potential.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.constants import si # TODO: this will change in the future

# Project
from streams.potential import *
from streams.integrate import leapfrog
from streams.coordinates import CartesianCoordinates

def plot_over_potential(xs, potential, grid=None):
    """ Plot the orbit of the particle(s) over potential contours. """

    if grid == None:
        # define the grid to compute the potential over
        grid = np.linspace(xs.min(), xs.max(), 200)

    fig, axes = potential.plot(grid,grid,grid)

    axes[0,0].plot(xs[:,:,0], xs[:,:,1], color="k")
    axes[1,0].plot(xs[:,:,0], xs[:,:,2], color="k")
    axes[1,1].plot(xs[:,:,1], xs[:,:,2], color="k")

    return fig,axes

def point_mass():
    G = u.Quantity(6.673E-11, u.m**3 / u.kg / u.s**2).to(u.au**3 / u.M_sun / u.yr**2).value

    def point_mass_model(params):
        ''' Mass in M_sun, distance in AU, time in yr '''
        f = lambda x,y,z: -G*params["M"] / np.sqrt(x**2 +y**2 + z**2)
        df_dx = lambda x,y,z: G*params["M"]*x / (x**2 + y**2 + z**2)**1.5
        df_dy = lambda x,y,z: G*params["M"]*y / (x**2 + y**2 + z**2)**1.5
        df_dz = lambda x,y,z: G*params["M"]*z / (x**2 + y**2 + z**2)**1.5
        return (f, df_dx, df_dy, df_dz)

    potential = Potential()
    potential.add_component("point_mass", point_mass_model({"M" : 1.}))

    initial_position = [1., 0., 0.1] # AU
    initial_velocity = [0.0, 2*np.pi, 0.] # AU/yr

    ts, xs, vs = integrate_potential(potential,
                        initial_position=initial_position,
                        initial_velocity=initial_velocity,
                        t1=0., t2=100., dt=0.01)
    make_plots(ts, xs, vs, "AU", "yr")

def binney_isochrone():
    """ TODO: description """

    G = si.G.to(u.kpc**3 / u.M_sun / u.Myr**2).value

    def isochrone_model(params):
        f = lambda x,y,z: -G*params["M"] / (params["b"] + np.sqrt(params["b"]**2 + x**2 + y**2 + z**2))
        df_dx = lambda x,y,z: G*params["M"]*x / (params["b"] + np.sqrt(params["b"]**2 + x**2 + y**2 + z**2/params["c"]**2))**2 / np.sqrt(params["b"]**2 + x**2 + y**2 + z**2/params["c"]**2)
        df_dy = lambda x,y,z: G*params["M"]*y / (params["b"] + np.sqrt(params["b"]**2 + x**2 + y**2 + z**2/params["c"]**2))**2 / np.sqrt(params["b"]**2 + x**2 + y**2 + z**2/params["c"]**2)
        df_dz = lambda x,y,z: G*params["M"]*z / (params["b"] + np.sqrt(params["b"]**2 + x**2 + y**2 + z**2/params["c"]**2))**2 / np.sqrt(params["b"]**2 + x**2 + y**2 + z**2/params["c"]**2)/params["c"]
        return (f, df_dx, df_dy, df_dz)

    isochrone_params = {"b" : 1., "M" : 1E12, "c" : 0.9}
    model_funcs = isochrone_model(isochrone_params)

    gal_potential = Potential(CartesianCoordinates)
    gal_potential.add_component("isochrone", model_funcs[0], derivs=model_funcs[1:])

    initial_position = np.array([1., 0., 0.2]) # kpc
    r02 = np.sum(initial_position**2)
    a = np.sqrt(isochrone_params["b"] + r02)
    vc = G*isochrone_params["M"]*r02 / (isochrone_params["b"]+a)**2 / a

    initial_velocity = [0.,
                        vc,
                        0.]

    ts, xs, vs = leapfrog(gal_potential.acceleration_at,
                          initial_position=initial_position,
                          initial_velocity=initial_velocity,
                          t1=0., t2=100., dt=0.1)

    fig, axes = plot_over_potential(xs, gal_potential)
    plt.show()

def three_component_galaxy():
    """ TODO: description """

    disk_potential = MiyamotoNagaiPotential(M=1E11*u.M_sun,
                                            a=6.5,
                                            b=0.26)
    bulge_potential = HernquistPotential(M=3.4E10*u.M_sun,
                                         c=0.7)
    halo_potential = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value,
                                            q1=1.38,
                                            q2=1.0,
                                            qz=1.36,
                                            phi=1.692969,
                                            c=12.)
    galaxy_potential = disk_potential + bulge_potential + halo_potential

    initial_position = np.array([10., 0., 0.2]) # kpc
    initial_velocity = np.array([150., 120., 15.]) # km/s
    initial_velocity = [(v*u.km/u.s).to(u.kpc/u.Myr).value for v in initial_velocity]

    ts, xs, vs = leapfrog(galaxy_potential.acceleration_at,
                          initial_position=initial_position,
                          initial_velocity=initial_velocity,
                          t1=0., t2=6000., dt=1.)

    fig, axes = plot_over_potential(xs, galaxy_potential)
    plt.show()

if __name__ == "__main__":
    #binney_isochrone()
    three_component_galaxy()