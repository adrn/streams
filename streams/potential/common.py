# coding: utf-8

""" Common Galactic potential components. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import uuid
import math

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

from .core import CartesianPotential

#import _common # minimal gains from using Cython

__all__ = ["PointMassPotential", "MiyamotoNagaiPotential",\
           "HernquistPotential", "LogarithmicPotentialLJ",\
           "PlummerPotential", "IsochronePotential",\
           "AxisymmetricNFWPotential", "AxisymmetricLogarithmicPotential"]

############################################################
#    Potential due to a point mass at a given position
#
def _cartesian_point_mass_model(bases):
    """ Generates functions to evaluate a point mass potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            m : mass
            r_0 : position of the point mass
    """

    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value

    def f(r,r_0,m):
        R = np.sqrt(np.sum((r-r_0)**2, axis=0))
        return -_G * m / R

    def df(r,r_0,m):
        a = (np.sum((r-r_0)**2, axis=0)**-1.5)
        return -_G * m * (r-r_0) * a

    return (f, df)

class PointMassPotential(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents a point-mass potential at the given origin.

            $\Phi = -\frac{GM}{r-r_0}$

            The parameters dictionary should include:
                r_0 : location of the point mass
                m : mass

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.

        """

        latex = "$\\Phi = -\\frac{GM}{r-r_0}$"

        assert "m" in parameters.keys(), "You must specify a mass."

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_point_mass_model(units)
        super(PointMassPotential, self).__init__(units,
                                                 f=f, f_prime=df,
                                                 latex=latex,
                                                 parameters=parameters)


##############################################################################
#    Miyamoto-Nagai Disk potential from Miyamoto & Nagai 1975
#    http://adsabs.harvard.edu/abs/1975PASJ...27..533M
#
def _cartesian_miyamoto_nagai_model(bases):
    """ Generates functions to evaluate a Miyamoto-Nagai potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            m : total mass in the potential
            a :
            b :
    """
    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value

    def f(r,r_0,m,a,b):
        rr = r-r_0
        x,y,z = rr

        z_term = a + np.sqrt(z*z + b*b)
        return -_G * m / np.sqrt(x*x + y*y + z_term*z_term)

    def df(r,r_0,m,a,b):
        rr = r-r_0
        x,y,z = rr

        sqrtz = np.sqrt(z*z + b*b)
        z_term = a + sqrtz
        fac = -_G*m*(x*x + y*y + z_term*z_term)**-1.5

        dx = fac*x
        dy = fac*y

        c = a / sqrtz
        dz = fac*z * (1. + c)

        return np.array([dx,dy,dz])

    return (f, df)

class MiyamotoNagaiPotential(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents the Miyamoto-Nagai potential (1975) for a disk-like
            potential.

            $\Phi_{disk} = -\frac{GM_{disk}}{\sqrt{R^2 + (a + sqrt{z^2 + b^2})^2}}$

            The parameters dictionary should include:
                r_0 : location of the origin
                m : mass in the potential
                a :
                b :

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.

        """

        latex = "$\\Phi_{disk} = -\\frac{GM_{disk}}{\\sqrt{R^2 + (a + \\sqrt{z^2 + b^2})^2}}$"

        assert "m" in parameters.keys(), "You must specify a mass."
        assert "a" in parameters.keys(), "You must specify the parameter 'a'."
        assert "b" in parameters.keys(), "You must specify the parameter 'b'."

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_miyamoto_nagai_model(units)
        super(MiyamotoNagaiPotential, self).__init__(units,
                                                     f=f, f_prime=df,
                                                     latex=latex,
                                                     parameters=parameters)

##############################################################################
#    Hernquist Spheroid potential from Hernquist 1990
#    http://adsabs.harvard.edu/abs/1990ApJ...356..359H
#
def _cartesian_hernquist_model(bases):
    """ Generates functions to evaluate a Hernquist potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            mass : total mass in the potential
            c : core/concentration parameter
    """

    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value

    def f(r,r_0,m,c):
        rr = r-r_0
        R = np.sqrt(np.sum(rr**2, axis=0))

        val = -_G * m / (R + c)
        return val

    def df(r,r_0,m,c):
        rr = r-r_0
        R = np.sqrt(np.sum(rr**2, axis=0))

        fac = -_G*m / ((R + c)**2 * R)
        return fac*rr

    return (f, df)

class HernquistPotential(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents the Hernquist potential (1990) for a spheroid (bulge).

            $\Phi_{spher} = -\frac{GM_{spher}}{r + c}$

            The parameters dictionary should include:
                r_0 : location of the origin
                m : mass in the potential
                c : core concentration

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.

        """


        latex = "$\\Phi_{spher} = -\\frac{GM_{spher}}{r + c}$"

        assert "m" in parameters.keys(), "You must specify a mass."
        assert "c" in parameters.keys(), "You must specify the parameter 'c'."

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_hernquist_model(units)
        super(HernquistPotential, self).__init__(units,
                                                 f=f, f_prime=df,
                                                 latex=latex,
                                                 parameters=parameters)

##############################################################################
#    Isochrone potential
#
def _cartesian_isochrone_model(bases):
    """ Generates functions to evaluate an Isochrone potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            m : total mass in the potential
            b : core/concentration parameter
    """

    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value

    def f(r,r_0,m,b):
        rr = np.sqrt(np.sum((r-r_0)**2, axis=0))
        val = -_G * m / (np.sqrt(rr**2 + b**2) + b)
        return val

    def df(r,r_0,m,b):
        rr = r-r_0
        R = np.sqrt(np.sum((rr)**2, axis=0))

        fac = -_G*m / (np.sqrt(R**2 + b**2) + b)
        return fac*rr

    return (f, df)

class IsochronePotential(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents the Isochrone potential.

            $\Phi_{spher} = -\frac{GM}{\sqrt{r^2+b^2} + b}$

            The parameters dictionary should include:
                r_0 : location of the origin
                m : mass in the potential
                b : core concentration

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.

        """


        latex = "$\\Phi = -\\frac{GM}{\sqrt{r^2+b^2} + b}$"

        assert "m" in parameters.keys(), "You must specify a mass."
        assert "b" in parameters.keys(), "You must specify the parameter 'b'."

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_isochrone_model(units)
        super(IsochronePotential, self).__init__(units,
                                                 f=f, f_prime=df,
                                                 latex=latex,
                                                 parameters=parameters)

##############################################################################
#    Plummer potential
#
def _cartesian_plummer_model(bases):
    """ Generates functions to evaluate a Plummer potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            mass : total mass in the potential
            a : core/concentration parameter
    """

    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value

    def f(r,r_0,m,a):
        try:
            rr = np.sqrt(np.sum((r-r_0)**2, axis=0))[:,np.newaxis]
        except IndexError:
            rr = np.sqrt(np.sum((r-r_0)**2, axis=0))
        val = -_G * m / np.sqrt(rr**2 + a**2)
        return val

    def df(r,r_0,m,a):
        rr = r-r_0
        try:
            R_sq = np.sum((rr)**2, axis=0)[:,np.newaxis]
        except IndexError:
            R_sq = np.sum((rr)**2, axis=0)

        fac = -_G*m / (R_sq + a**2)**1.5
        return fac*rr

    return (f, df)

class PlummerPotential(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents the Plummer potential

            $\Phi = -\frac{GM}{\sqrt{r^2 + a^2}}$

            The parameters dictionary should include:
                r_0 : location of the origin
                m : mass in the potential
                a : core concentration

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.

        """

        latex = r"$\Phi = -\frac{GM}{\sqrt{r^2 + a^2}}$"

        assert "m" in parameters.keys(), "You must specify a mass."
        assert "a" in parameters.keys(), "You must specify the parameter 'a'."

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_plummer_model(units)
        super(PlummerPotential, self).__init__(units,
                                                 f=f, f_prime=df,
                                                 latex=latex,
                                                 parameters=parameters)

##############################################################################
#    Triaxial, Logarithmic potential (see: Johnston et al. 1998)
#    http://adsabs.harvard.edu/abs/1999ApJ...512L.109J
#
def _cartesian_logarithmic_lj_model(bases):
    """ Generates functions to evaluate a Logarithmic potential and its
        derivatives at a specified position. This form of the log potential
        allows for an angle orientation in the X-Y plane.

        Physical parameters for this potential are:
            q1 : x-axis flattening parameter
            q2 : y-axis flattening parameter
            qz : z-axis flattening parameter
            phi : orientation angle in X-Y plane
            v_halo : circular velocity of the halo
            R_halo : radial concentration
    """

    def f(r,r_0,v_halo,q1,q2,qz,phi,R_halo):
        C1 = (math.cos(phi)/q1)**2+(math.sin(phi)/q2)**2
        C2 = (math.cos(phi)/q2)**2+(math.sin(phi)/q1)**2
        C3 = 2.*math.sin(phi)*math.cos(phi)*(1./q1**2 - 1./q2**2)

        rr = r-r_0
        x,y,z = rr

        return v_halo*v_halo * np.log(C1*x*x + C2*y*y + C3*x*y + z*z/qz**2 + R_halo**2)

    def df(r,r_0,v_halo,q1,q2,qz,phi,R_halo):
        C1 = (math.cos(phi)/q1)**2+(math.sin(phi)/q2)**2
        C2 = (math.cos(phi)/q2)**2+(math.sin(phi)/q1)**2
        C3 = 2.*math.sin(phi)*math.cos(phi)*(1./q1**2 - 1./q2**2)

        rr = r-r_0
        x,y,z = rr

        fac = v_halo*v_halo / (C1*x*x + C2*y*y + C3*x*y + z*z/qz**2 + R_halo**2)

        dx = fac * (2.*C1*x + C3*y)
        dy = fac * (2.*C2*y + C3*x)
        dz = 2.* fac * z * qz**-2

        return -np.array([dx,dy,dz])

    return (f, df)

class LogarithmicPotentialLJ(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents a triaxial Logarithmic potential (e.g. triaxial halo).

            $\Phi_{halo} = v_{halo}^2\ln(C1x^2 + C2y^2 + C3xy + z^2/q_z^2 + R_halo^2)$

            Model parameters: v_halo, q1, q2, qz, phi, R_halo, origin

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.
        """

        latex = "$\\Phi_{halo} = v_{halo}^2\\ln(C_1x^2 + C_2y^2 + C_3xy + z^2/q_z^2 + R_halo^2)$"

        for p in ["q1", "q2", "qz", "phi", "v_halo", "R_halo"]:
            assert p in parameters.keys(), \
                    "You must specify the parameter '{0}'.".format(p)

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_logarithmic_lj_model(units)
        super(LogarithmicPotentialLJ, self).__init__(units,
                                                     f=f, f_prime=df,
                                                     latex=latex,
                                                     parameters=parameters)


##############################################################################
#    Axisymmetric NFW potential
#
def _cartesian_axisymmetric_nfw_model(bases):
    """ Generates functions to evaluate an NFW potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            m : total mass in the potential
            qz : z axis flattening
            Rs : scale-length
    """

    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value

    def f(r,r_0,log_m,qz,Rs):
        rr = r-r_0
        x,y,z = rr

        m = np.exp(log_m)
        R_sq = x**2 + y**2
        sqrt_term = np.sqrt(R_sq + (z/qz)**2)
        val = -_G * m / sqrt_term * np.log(1. + sqrt_term/Rs)

        return val

    def df(r,r_0,log_m,qz,Rs):
        rr = r-r_0
        x,y,z = rr

        m = np.exp(log_m)
        zz = z/qz
        R = np.sqrt(x*x + y*y + zz*zz)

        term1 = 1./(R*R*(Rs+R))
        term2 = -np.log(1. + R/Rs) / (R*R*R)
        fac = _G*m * (term1 + term2)
        _x = fac * x
        _y = fac * y
        _z = fac * z / qz**2

        return np.array([_x,_y,_z])

    return (f, df)

class AxisymmetricNFWPotential(CartesianPotential):

    def __init__(self, units, **parameters):

        latex = "$\sigma$"

        assert "log_m" in parameters.keys(), "You must specify a log-mass."
        assert "qz" in parameters.keys(), "You must specify the parameter 'qz'."
        assert "Rs" in parameters.keys(), "You must specify the parameter 'Rs'."

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_axisymmetric_nfw_model(units)
        super(AxisymmetricNFWPotential, self).__init__(units,
                                                 f=f, f_prime=df,
                                                 latex=latex,
                                                 parameters=parameters)

##############################################################################
#    Axisymmetric, Logarithmic potential
#
def _cartesian_axisymmetric_logarithmic_model(bases):
    """ Generates functions to evaluate a Logarithmic potential and its
        derivatives at a specified position.

        Physical parameters for this potential are:
            qz : z-axis flattening parameter
            v_c : circular velocity of the halo
    """

    def f(r,r_0,v_c,qz):
        rr = r-r_0
        x,y,z = rr

        return 0.5*v_c*v_c * np.log(x*x + y*y + z*z/qz**2)

    def df(r,r_0,v_c,qz):
        rr = r-r_0
        x,y,z = rr

        fac = v_c*v_c / (x*x + y*y + z*z/(qz*qz))

        dx = fac * x
        dy = fac * y
        dz = fac * z / (qz*qz)

        return -np.array([dx,dy,dz]).T

    return (f, df)

class AxisymmetricLogarithmicPotential(CartesianPotential):

    def __init__(self, units, **parameters):
        """ Represents an axisymmetric Logarithmic potential

            $\Phi_{halo} = v_{c}^2/2\ln(x^2 + y^2 + z^2/q_z^2)$

            Model parameters: v_c, qz

            Parameters
            ----------
            units : list
                Defines a system of physical base units for the potential.
            parameters : dict
                A dictionary of parameters for the potential definition.
        """

        latex = "$\\Phi_{halo} = v_{halo}^2\\ln(C_1x^2 + C_2y^2 + C_3xy + z^2/q_z^2 + R_halo^2)$"

        for p in ["qz", "v_c"]:
            assert p in parameters.keys(), \
                    "You must specify the parameter '{0}'.".format(p)

        # get functions for evaluating potential and derivatives
        f,df = _cartesian_axisymmetric_logarithmic_model(units)
        super(AxisymmetricLogarithmicPotential, self).__init__(units,
                                                     f=f, f_prime=df,
                                                     latex=latex,
                                                     parameters=parameters)