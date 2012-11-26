# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import uuid

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants.si import G as _G

from .core import Potential
from ..util import *
#import _common # minimal gains from using Cython

__all__ = ["MiyamotoNagaiPotential", "HernquistPotential", "LogarithmicPotential", "LogarithmicPotentialJZSH", "LogarithmicPotentialJHB", "PointMassPotential"]

def _raise(ex):
    raise ex

####################################################################################
#    Potential due to a point mass at the given position
####################################################################################

def _point_mass_model_cartesian(params):
    ''' A function that accepts model parameters, and returns a tuple of functions that accept
        coordinates and evaluates this component of the potential and its derivatives for cartesian
        coordinates.
    '''
    f = lambda x,y,z: -params["_G"] * params["M"] / np.sqrt((x-params["x0"])**2 + (y-params["y0"])**2 + (z-params["z0"])**2)
    df_dx = lambda x,y,z: params["_G"] * params["M"]*(x-params["x0"]) / ((x-params["x0"])**2 + (y-params["y0"])**2 + (z-params["z0"])**2)**1.5
    df_dy = lambda x,y,z: params["_G"] * params["M"]*(y-params["y0"]) / ((x-params["x0"])**2 + (y-params["y0"])**2 + (z-params["z0"])**2)**1.5
    df_dz = lambda x,y,z: params["_G"] * params["M"]*(z-params["z0"]) / ((x-params["x0"])**2 + (y-params["y0"])**2 + (z-params["z0"])**2)**1.5
    return (f, df_dx, df_dy, df_dz)

class PointMassPotential(Potential):

    def __init__(self, M, location, coord_sys=cartesian3D, length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass):
        ''' Represents a point-mass potential at the given origin.

            $\Phi = -\frac{GM}{r-r_0}$

            Parameters
            ----------
            M : numeric or Quantity
                Mass.
            location : dict
                Must be a dictionary specifying the location of the point mass. For example, in cartesian, it
                should look like location = {'x0' : 5., 'y0' : 10., 'z0' : -10.}.

        '''
        super(PointMassPotential, self).__init__()
        self.length_unit = u.Unit(length_unit)
        self.time_unit = u.Unit(time_unit)
        self.mass_unit = u.Unit(mass_unit)

        # First see if M is a Quantity-like object
        if hasattr(M, 'to'):
            M_val = M.to(self.mass_unit).value
        else:
            M_val = float(M)

        self.params = {"M" : float(M_val),
                       "_G" : u.Quantity(_G, u.m**3 / u.kg / u.s**2).to(self.length_unit**3 / self.mass_unit / self.time_unit**2).value}

        # Trust that the user knows what they are doing with 'origin'
        for key,val in location.items():
            self.params[key] = val

        if coord_sys == cartesian3D:
            self.coordinate_system = cartesian3D
            f_derivs = _point_mass_model_cartesian(self.params)
        else:
            raise ValueError("'coord_sys' can be util.cartesian3D.")

        self.add_component(uuid.uuid4(), f_derivs[0], derivs=f_derivs[1:])
        self.add_component = lambda *args: _raise(AttributeError("'add_component' is only available for the base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi = -\\frac{GM}{r-r_0}$"

####################################################################################
#    Miyamoto-Nagai Disk potential from Miyamoto & Nagai 1975
#    http://adsabs.harvard.edu/abs/1975PASJ...27..533M
####################################################################################

def _miyamoto_nagai_model_cartesian(params):
    ''' A function that accepts model parameters, and returns a tuple of functions that accept
        coordinates and evaluates this component of the potential and its derivatives for cartesian
        coordinates.
    '''
    f = lambda x,y,z: -params["_G"] * params["M"] / np.sqrt(x**2 + y**2 + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)
    df_dx = lambda x,y,z: params["_G"] * params["M"]*x / ((x**2 + y**2) + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5
    df_dy = lambda x,y,z: params["_G"] * params["M"]*y / ((x**2 + y**2) + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5
    df_dz = lambda x,y,z: params["_G"] * params["M"]*z*(1 + params["a"]/np.sqrt(z**2 + params["b"]**2)) / ((x**2 + y**2) + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5

    #df_dx = lambda x,y,z: _common._miyamoto_nagai_dx(params["_G"], params["M"], params["a"], params["b"], x, y, z, len(x))
    #df_dy = lambda x,y,z: _common._miyamoto_nagai_dy(params["_G"], params["M"], params["a"], params["b"], x, y, z, len(x))
    #df_dz = lambda x,y,z: _common._miyamoto_nagai_dz(params["_G"], params["M"], params["a"], params["b"], x, y, z, len(x))
    return (f, df_dx, df_dy, df_dz)

def _miyamoto_nagai_model_cylindrical(params):
    ''' A function that accepts model parameters, and returns a tuple of functions that accept
        coordinates and evaluates this component of the potential and its derivatives for cylindrical
        coordinates.
    '''
    f = lambda R, phi, z: -params["_G"] * params["M"] / np.sqrt(R**2 + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)
    df_dR = lambda R, phi, z: params["_G"] * params["M"]*R / (R**2 + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5
    df_dphi = lambda R, phi, z: 0.
    df_dz = lambda R, phi, z: params["_G"] * params["M"]*z*(1 + params["a"]/np.sqrt(z**2 + params["b"]**2)) / (R**2 + (params["a"] + np.sqrt(z**2 + params["b"]**2))**2)**1.5
    return (f, df_dR, df_dphi, df_dz)

class MiyamotoNagaiPotential(Potential):

    def __init__(self, M, a, b, coord_sys=cartesian3D, length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass):
        ''' Represents the Miyamoto-Nagai potential (1975) for a disk-like potential.

            $\Phi_{disk} = -\frac{GM_{disk}}{\sqrt{R^2 + (a + sqrt{z^2 + b^2})^2}}$

        '''
        super(MiyamotoNagaiPotential, self).__init__()
        self.length_unit = u.Unit(length_unit)
        self.time_unit = u.Unit(time_unit)
        self.mass_unit = u.Unit(mass_unit)

        # First see if M is a Quantity-like object
        if hasattr(M, 'to'):
            M_val = M.to(self.mass_unit).value
        else:
            M_val = float(M)

        self.params = {"a" : float(a),
                       "b" : float(b),
                       "M" : float(M_val),
                       "_G" : u.Quantity(_G, u.m**3 / u.kg / u.s**2).to(self.length_unit**3 / self.mass_unit / self.time_unit**2).value}

        if coord_sys == cartesian3D:
            self.coordinate_system = cartesian3D
            f_derivs = _miyamoto_nagai_model_cartesian(self.params)
        elif coord_sys == cylindrical3D:
            self.coordinate_system = cylindrical3D
            f_derivs = _miyamoto_nagai_model_cylindrical(self.params)
        else:
            raise ValueError("'coord_sys' can be util.cartesian3D or util.cylindrical3D.")

        self.add_component(uuid.uuid4(), f_derivs[0], derivs=f_derivs[1:])
        self.add_component = lambda *args: _raise(AttributeError("'add_component' is only available for the base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{disk} = -\\frac{GM_{disk}}{\\sqrt{R^2 + (a + \\sqrt{z^2 + b^2})^2}}$"

####################################################################################
#    Hernquist Spheroid potential from Hernquist 1990
#    http://adsabs.harvard.edu/abs/1990ApJ...356..359H
####################################################################################

def _hernquist_model_cartesian(params):
    ''' A function that accepts model parameters, and returns a tuple of functions that accept
        coordinates and evaluates this component of the potential and its derivatives for cartesian
        coordinates.
    '''
    f = lambda x,y,z: - params["_G"]*params["M"] / (np.sqrt(x**2 + y**2 + z**2) + params["c"])
    df_dx = lambda x,y,z: params["_G"]*params["M"]*x / (np.sqrt(x**2 + y**2 + z**2) + params["c"])**2 / np.sqrt(x**2 + y**2 + z**2)
    df_dy = lambda x,y,z: params["_G"]*params["M"]*y / (np.sqrt(x**2 + y**2 + z**2) + params["c"])**2 / np.sqrt(x**2 + y**2 + z**2)
    df_dz = lambda x,y,z: params["_G"]*params["M"]*z / (np.sqrt(x**2 + y**2 + z**2) + params["c"])**2 / np.sqrt(x**2 + y**2 + z**2)
    return (f, df_dx, df_dy, df_dz)

def _hernquist_model_spherical(params):
    ''' A function that accepts model parameters, and returns a tuple of functions that accept
        coordinates and evaluates this component of the potential and its derivatives for spherical
        coordinates.
    '''
    f = lambda r,phi,theta: - params["_G"]*params["M"] / (r + params["c"])
    df_dr = lambda r,phi,theta: params["_G"]*params["M"] / (r + params["c"])**2
    df_dphi = lambda r,phi,theta: 0.
    df_dtheta = lambda r,phi,theta: 0.
    return (f, df_dr, df_dphi, df_dtheta)

class HernquistPotential(Potential):

    def __init__(self, M, c, coord_sys=cartesian3D, length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass):
        ''' Represents the Hernquist potential (1990) for a spheroid (bulge).

            $\Phi_{spher} = -\frac{GM_{spher}}{r + c}$

        '''
        super(HernquistPotential, self).__init__()
        self.length_unit = u.Unit(length_unit)
        self.time_unit = u.Unit(time_unit)
        self.mass_unit = u.Unit(mass_unit)

        # First see if M is a Quantity-like object
        if hasattr(M, 'to'):
            M_val = M.to(self.mass_unit).value
        else:
            M_val = float(M)

        self.params = {"c" : float(c),
                       "M" : float(M_val),
                       "_G" : u.Quantity(_G, u.m**3 / u.kg / u.s**2).to(self.length_unit**3 / self.mass_unit / self.time_unit**2).value}

        if coord_sys == cartesian3D:
            self.coordinate_system = cartesian3D
            f_derivs = _hernquist_model_cartesian(self.params)
        elif coord_sys == spherical3D:
            self.coordinate_system = spherical3D
            f_derivs = _hernquist_model_spherical(self.params)
        else:
            raise ValueError("'coord_sys' can be cartesian or spherical.")

        self.add_component(uuid.uuid4(), f_derivs[0], derivs=f_derivs[1:])
        self.add_component = lambda *args: _raise(AttributeError("'add_component' is only available for the base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{spher} = -\\frac{GM_{spher}}{r + c}$"

####################################################################################
#    Triaxial, Logarithmic potential (see: Johnston et al. 1998)
#    http://adsabs.harvard.edu/abs/1999ApJ...512L.109J
####################################################################################

def _logarithmic_model_cartesian_jzsh(params):
    ''' A function that accepts model parameters, and returns a tuple of functions that accept
        coordinates and evaluates this component of the potential and its derivatives for cartesian
        coordinates.
    '''
    f = lambda x,y,z: params["v_circ"]**2 / 2. * np.log(x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["c"]**2)
    df_dx = lambda x,y,z: params["v_circ"]**2 * x / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["c"]**2)
    df_dy = lambda x,y,z: params["v_circ"]**2 * y / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["c"]**2) / params["p"]**2
    df_dz = lambda x,y,z: params["v_circ"]**2 * z / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["c"]**2) / params["q"]**2
    return (f, df_dx, df_dy, df_dz)

class LogarithmicPotentialJZSH(Potential):

    def __init__(self, v_circ, c, p, q, coord_sys=cartesian3D, length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass):
        ''' Represents a triaxial Logarithmic potential (e.g. triaxial halo).

            $\Phi_{halo} = \frac{v_{circ}^2}{2}\ln(x^2 + y^2/p^2 + z^2/q^2 + c^2)$

        '''
        super(LogarithmicPotentialJZSH, self).__init__()
        self.length_unit = u.Unit(length_unit)
        self.time_unit = u.Unit(time_unit)
        self.mass_unit = u.Unit(mass_unit)

        # First see if v_circ is a Quantity-like object
        if hasattr(v_circ, 'to'):
            v_circ_val = v_circ.to(self.length_unit / self.time_unit).value
        else:
            v_circ_val = float(v_circ)

        self.params = {"c" : float(c),
                       "p" : float(p),
                       "q" : float(q),
                       "v_circ" : float(v_circ_val)}

        if coord_sys == cartesian3D:
            self.coordinate_system = cartesian3D
            f_derivs = _logarithmic_model_cartesian(self.params)
        else:
            raise ValueError("'coord_sys' can only be cartesian.")

        self.add_component(uuid.uuid4(), f_derivs[0], derivs=f_derivs[1:])
        self.add_component = lambda *args: _raise(AttributeError("'add_component' is only available for the base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{halo} = \\frac{v_{circ}^2}{2}\\ln(x^2 + y^2/p^2 + z^2/q^2 + c^2)$"

LogarithmicPotential = LogarithmicPotentialJZSH

def _logarithmic_model_cartesian_jhb(params):
    ''' A function that accepts model parameters, and returns a tuple of functions that accept
        coordinates and evaluates this component of the potential and its derivatives for cartesian
        coordinates.
    '''
    f = lambda x,y,z: params["v_halo"]**2 * np.log(x**2 + y**2 + z**2 + params["d"]**2)
    df_dx = lambda x,y,z: params["v_halo"]**2 * 2 * x / (x**2 + y**2 + z**2 + params["d"]**2)
    df_dy = lambda x,y,z: params["v_halo"]**2 * 2 * y / (x**2 + y**2 + z**2 + params["d"]**2)
    df_dz = lambda x,y,z: params["v_halo"]**2 * 2 * z / (x**2 + y**2 + z**2 + params["d"]**2)
    return (f, df_dx, df_dy, df_dz)

class LogarithmicPotentialJHB(Potential):

    def __init__(self, v_halo, d, coord_sys=cartesian3D, length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass):
        ''' Represents a triaxial Logarithmic potential (e.g. triaxial halo).

            $\Phi_{halo} = \frac{v_{circ}^2}{2}\ln(x^2 + y^2/p^2 + z^2/q^2 + c^2)$

        '''
        super(LogarithmicPotentialJHB, self).__init__()
        self.length_unit = u.Unit(length_unit)
        self.time_unit = u.Unit(time_unit)
        self.mass_unit = u.Unit(mass_unit)

        # First see if v_halo is a Quantity-like object
        if hasattr(v_halo, 'to'):
            v_halo_val = v_halo.to(self.length_unit / self.time_unit).value
        else:
            v_halo_val = float(v_halo)

        self.params = {"d" : float(d),
                       "v_halo" : float(v_halo_val)}

        if coord_sys == cartesian3D:
            self.coordinate_system = cartesian3D
            f_derivs = _logarithmic_model_cartesian_jhb(self.params)
        else:
            raise ValueError("'coord_sys' can only be cartesian.")

        self.add_component(uuid.uuid4(), f_derivs[0], derivs=f_derivs[1:])
        self.add_component = lambda *args: _raise(AttributeError("'add_component' is only available for the base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{halo} = v_{halo}^2\\ln(x^2 + y^2 + z^2 + d^2)$"

