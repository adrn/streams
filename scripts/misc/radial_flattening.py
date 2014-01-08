# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import astropy.units as u

# Project
from streams import usys
from streams.potential import CartesianPotential
from streams.integrate import LeapfrogIntegrator

# Create logger
logger = logging.getLogger(__name__)

##############################################################################
#    Axisymmetric, Logarithmic potential
#
def _axisymmetric_logarithmic_model(bases):
    """ Generates functions to evaluate a Logarithmic potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            s : inverse of z-axis flattening parameter
            a : factor outside of logarithm
            b : core radius
    """

    def f(r,r_0,s,a,b):
        rr = r-r_0
        x,y,z = rr.T

        return a*np.log(x**2 + y**2 + (s*z)**2 + b**2)

    def df(r,r_0,s,a,b):
        rr = r-r_0
        x,y,z = rr.T

        fac = a / (x**2 + y**2 + (s*z)**2 + b**2)

        dx = fac * (2*x)
        dy = fac * (2*y)
        dz = fac * 2 * s**2 * z

        return -np.array([dx,dy,dz]).T

    return (f, df)

class AxisymmetricLogPotential(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents an axisymmetric Logarithmic potential

            $\Phi = a\ln(R^2 + s^2z^2 + b^2)$

            Model parameters: s,a,b

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.
        """

        latex = "$\\Phi = a\\ln(R^2 + (sz)^2 + b^2)$"

        for p in ["s", "a", "b"]:
            assert p in parameters.keys(), \
                    "You must specify the parameter '{0}'.".format(p)

        # get functions for evaluating potential and derivatives
        f,df = _axisymmetric_logarithmic_model(units)
        super(AxisymmetricLogPotential, self).__init__(units,
                                                     f=f, f_prime=df,
                                                     latex=latex,
                                                     parameters=parameters)

##############################################################################
#    Axisymmetric, Logarithmic potential
#
def _axisymmetric_log_radial_flattening_model(bases):
    """ Generates functions to evaluate a Logarithmic potential with a distance-
        dependent z flattening and its derivatives at a specified position.

        Physical parameters for this potential are:

            a : factor outside of logarithm
            b : core radius
    """

    def f(r,r_0,a,b):
        rr = r-r_0
        x,y,z = rr.T

        r = np.sqrt(x**2 + y**2 + z**2)
        s = 1. + 36./(r+5)**2

        return a*np.log(x**2 + y**2 + (s*z)**2 + b**2)

    def df(r,r_0,a,b):
        rr = r-r_0
        x,y,z = rr.T

        fac = a / (x**2 + y**2 + (s*z)**2 + b**2)

        dx = fac * (2*x)
        dy = fac * (2*y)
        dz = fac * 2 * s**2 * z

        return -np.array([dx,dy,dz]).T

    return (f, df)

class AxisymmetricLogRadialFlatteningPotential(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents an axisymmetric Logarithmic potential

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.
        """

        latex = "$\\Phi = a\\ln(R^2 + (sz)^2 + b^2)$"

        for p in ["a", "b"]:
            assert p in parameters.keys(), \
                    "You must specify the parameter '{0}'.".format(p)

        # get functions for evaluating potential and derivatives
        f,df = _axisymmetric_log_radial_flattening_model(units)
        super(AxisymmetricLogRadialFlatteningPotential, self).__init__(units,
                                                     f=f, f_prime=df,
                                                     latex=latex,
                                                     parameters=parameters)

def plot_contours():
    potential1 = AxisymmetricLogPotential(units=usys,
                                         s=1.,
                                         a=0.5*(u.km/u.s)**2,
                                         b=2.*u.kpc)
    potential2 = AxisymmetricLogRadialFlatteningPotential(units=usys,
                                         a=0.5*(u.km/u.s)**2,
                                         b=2.*u.kpc)
    # potential2 = AxisymmetricLogPotential(units=usys,
    #                                      s=2.,
    #                                      a=0.5*(u.km/u.s)**2,
    #                                      b=0.)

    grid = np.linspace(-150, 150, 200)*u.kpc
    fig1 = potential1.plot(ndim=3, grid=grid)
    fig2 = potential2.plot(ndim=3, grid=grid)

    plt.show()

def main():

    potential = AxisymmetricLogPotential(units=usys,
                                         s=1/0.9,
                                         a=0.5*(u.kpc/u.Myr)**2,
                                         b=0.*u.kpc)

    x0 = np.array([10.0, 0.0, 2.]) # kpc
    v0 = (np.array([0., 220., 20.])*u.km/u.s)\
                    .to(u.kpc/u.Myr).value # kpc/Myr

    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    x0.T, v0.T)
    ts, xs, vs = integrator.run(dt=0.1, Nsteps=10000)

    fig,axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,10))
    axes[0,0].plot(xs[:,0,0], xs[:,0,1])
    axes[0,0].set_xlim(-12,12)
    axes[0,0].set_ylim(-12,12)

    axes[1,0].plot(xs[:,0,0], xs[:,0,2])

    axes[1,1].plot(xs[:,0,1], xs[:,0,2])
    axes[0,1].set_visible(False)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()

if __name__ == "__main__":
    plot_contours()
    #main()