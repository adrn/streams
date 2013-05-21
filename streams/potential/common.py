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

__all__ = ["PointMassPotential"]

#, "MiyamotoNagaiPotential", "HernquistPotential", "LogarithmicPotentialLJ"]

############################################################
#    Potential due to a point mass at a given position
#
def _cartesian_point_mass_model(bases):
    """ Generates functions to evaluate a point mass potential and its 
        derivatives at a specified position.
        
        Physical parameters for this potential are:
            mass
    """
    
    # scale G to be in this unit system
    _G = G.decompose(bases=bases).value
    
    def f(r,r_0,m): 
        return -_G * m / np.sqrt(np.sum((r-r_0)**2, axis=-1))
    
    def df(r,r_0,m):
        try:
            a = (np.sum((r-r_0)**2, axis=-1)**-1.5)[:,np.newaxis]
        except IndexError:
            a = (np.sum((r-r_0)**2, axis=-1)**-1.5)
            
        return -_G * m * (r-r_0) * a
        
    return (f, df)

class PointMassPotential(CartesianPotential):

    def __init__(self, unit_system, **parameters):
        """ Represents a point-mass potential at the given origin.

            $\Phi = -\frac{GM}{r-r_0}$
            
            The parameters dictionary should include:
                r_0 : location of the point mass
                m : mass
            
            Parameters
            ----------
            units : list, iterable
                A list of astropy Unit objects that specify the base units
                for length, time, mass, etc.
            parameters : dict
                A dictionary of parameters for the potential definition.

        """                
        
        latex = "$\\Phi = -\\frac{GM}{r-r_0}$"
        
        assert "m" in parameters.keys(), "You must specify a mass."
        assert "r_0" in parameters.keys(), "You must specify a location for the mass."
        
        unit_system = self._validate_unit_system(unit_system)
        
        # get functions for evaluating potential and derivatives
        f,df = _cartesian_point_mass_model(unit_system.bases)
        super(PointMassPotential, self).__init__(unit_system, 
                                                 f=f, f_prime=df, 
                                                 latex=latex, 
                                                 parameters=parameters)


'''


####################################################################################
#    Miyamoto-Nagai Disk potential from Miyamoto & Nagai 1975
#    http://adsabs.harvard.edu/abs/1975PASJ...27..533M
####################################################################################

def _cartesian_miyamoto_nagai_model(bases):
    """ Generates functions to evaluate a Miyamoto-Nagai potential and its 
        derivatives at a specified position.
        
        Physical parameters for this potential are:
            mass : total mass in the potential
            a : TODO
            b : TODO
    """
    # scale G to be in this unit system
    _G = G.decompose(bases=bases.values()).value
    
    def f(x,y,z,origin,m,a,b): 
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        return -_G * m / np.sqrt(xx**2 + yy**2 + (a + np.sqrt(zz**2 + b**2))**2)
    
    def df(x,y,z,origin,m,a,b): 
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        
        fac = _G*m*((xx**2 + yy**2) + (a + np.sqrt(zz**2 + b**2))**2)**-1.5
                
        dx = fac*xx
        dy = fac*yy
        _tmp = a/(np.sqrt(zz**2 + b**2))
        dz = fac*zz * (1.+_tmp)
        
        return np.array([dx,dy,dz])
        
    return (f, df)

class MiyamotoNagaiPotential(CartesianPotential):

    def __init__(self, units, origin=[0.,0.,0.]*u.kpc, **parameters):
        """ Represents the Miyamoto-Nagai potential (1975) for a disk-like
            potential.

            $\Phi_{disk} = -\frac{GM_{disk}}{\sqrt{R^2 + (a + sqrt{z^2 + b^2})^2}}$

        """
        
        latex = "$\\Phi_{disk} = -\\frac{GM_{disk}}{\\sqrt{R^2 + (a + \\sqrt{z^2 + b^2})^2}}$"
        
        assert "m" in parameters.keys(), "You must specify a mass."
        assert "a" in parameters.keys(), "You must specify the parameter 'a'."
        assert "b" in parameters.keys(), "You must specify the parameter 'b'."
        
        # get functions for evaluating potential and derivatives
        units = self._validate_unit_system(units)
        f,df = _cartesian_miyamoto_nagai_model(units)
        super(MiyamotoNagaiPotential, self).__init__(units, f=f, f_prime=df, 
                                                 latex=latex, origin=origin,
                                                 parameters=parameters)

####################################################################################
#    Hernquist Spheroid potential from Hernquist 1990
#    http://adsabs.harvard.edu/abs/1990ApJ...356..359H
####################################################################################

def _cartesian_hernquist_model(bases):
    """ Generates functions to evaluate a Hernquist potential and its 
        derivatives at a specified position.
        
        Physical parameters for this potential are:
            mass : total mass in the potential
            c : core/concentration parameter
    """
    
    # scale G to be in this unit system
    _G = G.decompose(bases=bases.values()).value
    
    def f(x,y,z,origin,m,c): 
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        val = -_G * m / (np.sqrt(xx**2 + yy**2 + zz**2) + c)
        return val
    
    def df(x,y,z,origin,m,c):
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        
        fac = _G*m / ((np.sqrt(xx**2 + yy**2 + zz**2) + c)**2 * np.sqrt(xx**2 + yy**2 + zz**2))
        
        dx = fac*xx
        dy = fac*yy
        dz = fac*zz
        
        return np.array([dx,dy,dz])
        
    return (f, df)
    
class HernquistPotential(CartesianPotential):
    
    def __init__(self, units, origin=[0.,0.,0.]*u.kpc, **parameters):
        """ Represents the Hernquist potential (1990) for a spheroid (bulge).

            $\Phi_{spher} = -\frac{GM_{spher}}{r + c}$
        """
        
        latex = "$\\Phi_{spher} = -\\frac{GM_{spher}}{r + c}$"
        
        assert "m" in parameters.keys(), "You must specify a mass."
        assert "c" in parameters.keys(), "You must specify the parameter 'c'."
        
        # get functions for evaluating potential and derivatives
        units = self._validate_unit_system(units)
        f,df = _cartesian_hernquist_model(units)
        super(HernquistPotential, self).__init__(units, f=f, f_prime=df, 
                                                 latex=latex, origin=origin,
                                                 parameters=parameters)

####################################################################################
#    Triaxial, Logarithmic potential (see: Johnston et al. 1998)
#    http://adsabs.harvard.edu/abs/1999ApJ...512L.109J
####################################################################################

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
            r_halo : radial concentration
    """

    def f(x,y,z,v_halo,q1,q2,qz,phi,r_halo,origin): 
        C1 = (math.cos(phi)/q1)**2+(math.sin(phi)/q2)**2
        C2 = (math.cos(phi)/q2)**2+(math.sin(phi)/q1)**2
        C3 = 2.*math.sin(phi)*math.cos(phi)*(1./q1**2 - 1./q2**2)
        
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        
        return v_halo**2 * np.log(C1*xx**2 + C2*yy**2 + C3*xx*yy + zz**2/qz**2 + r_halo**2)
    
    def df(x,y,z,v_halo,q1,q2,qz,phi,r_halo,origin):
        C1 = (math.cos(phi)/q1)**2+(math.sin(phi)/q2)**2
        C2 = (math.cos(phi)/q2)**2+(math.sin(phi)/q1)**2
        C3 = 2.*math.sin(phi)*math.cos(phi)*(1./q1**2 - 1./q2**2)
        
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        
        fac = v_halo**2 / (C1*xx**2 + C2*yy**2 + C3*xx*yy + zz**2/qz**2 + r_halo**2)
        
        dx = fac * (2.*C1*xx + C3*yy)
        dy = fac * (2.*C2*yy + C3*xx)
        dz = 2.* fac * zz * qz**-2
        
        return np.array([dx,dy,dz])
        
    return (f, df)

class LogarithmicPotentialLJ(CartesianPotential):

    def __init__(self, units, origin=[0.,0.,0.]*u.kpc, **parameters):
        """ Represents a triaxial Logarithmic potential (e.g. triaxial halo).
            
            $\Phi_{halo} = v_{halo}^2\ln(C1x^2 + C2y^2 + C3xy + z^2/q_z^2 + r_halo^2)$
            
            Parameters: v_halo, q1, q2, qz, phi, r_halo, origin

        """
        
        latex = "$\\Phi_{halo} = v_{halo}^2\\ln(C_1x^2 + C_2y^2 + C_3xy + z^2/q_z^2 + r_halo^2)$"
        
        for p in ["q1", "q2", "qz", "phi", "v_halo", "r_halo"]:
            assert p in parameters.keys(), \
                    "You must specify the parameter '{0}'.".format(p)
        
        # get functions for evaluating potential and derivatives
        units = self._validate_unit_system(units)
        f,df = _cartesian_logarithmic_lj_model(units)
        super(LogarithmicPotentialLJ, self).__init__(units, f=f, f_prime=df, 
                                                 latex=latex, origin=origin,
                                                 parameters=parameters)

'''

"""

def _logarithmic_model_cartesian_jzsh(params):
    ''' A function that accepts model parameters, and returns a tuple of
        functions that accept coordinates and evaluates this component of the
        potential and its derivatives for cartesian coordinates.
    '''
    f = lambda x,y,z: params["v_circ"]**2 / 2. * np.log(x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["c"]**2)
    df_dx = lambda x,y,z: params["v_circ"]**2 * x / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["c"]**2)
    df_dy = lambda x,y,z: params["v_circ"]**2 * y / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["c"]**2) / params["p"]**2
    df_dz = lambda x,y,z: params["v_circ"]**2 * z / (x**2 + y**2/params["p"]**2 + z**2/params["q"]**2 + params["c"]**2) / params["q"]**2
    return (f, df_dx, df_dy, df_dz)

class LogarithmicPotentialJZSH(Potential):

    def __init__(self, v_circ, c, p, q, coordinate_system=CartesianCoordinates,\
                 length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass):
        ''' Represents a triaxial Logarithmic potential (e.g. triaxial halo).

            $\Phi_{halo} = \frac{v_{circ}^2}{2}\ln(x^2 + y^2/p^2 + z^2/q^2 + c^2)$

        '''
        super(LogarithmicPotentialJZSH, self).\
            __init__(coordinate_system=coordinate_system)
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

        if coordinate_system == CartesianCoordinates:
            f_derivs = _logarithmic_model_cartesian_jzsh(self.params)
        else:
            raise ValueError("'coordinate_system' can only be cartesian.")

        self.coordinate_system = coordinate_system
        self.add_component(uuid.uuid4(), f_derivs[0], derivs=f_derivs[1:])
        self.add_component = lambda *args: \
            _raise(AttributeError("'add_component' is only available for the "
                                  "base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{halo} = \\frac{v_{circ}^2}{2}\\ln(x^2 + y^2/p^2 + z^2/q^2 + c^2)$"

LogarithmicPotential = LogarithmicPotentialJZSH

def _logarithmic_model_cartesian_jhb(params):
    ''' A function that accepts model parameters, and returns a tuple of
        functions that accept coordinates and evaluates this component of the
        potential and its derivatives for cartesian coordinates.
    '''
    f = lambda x,y,z: params["v_halo"]**2 * np.log(x**2 + y**2 + z**2 + params["d"]**2)
    df_dx = lambda x,y,z: params["v_halo"]**2 * 2 * x / (x**2 + y**2 + z**2 + params["d"]**2)
    df_dy = lambda x,y,z: params["v_halo"]**2 * 2 * y / (x**2 + y**2 + z**2 + params["d"]**2)
    df_dz = lambda x,y,z: params["v_halo"]**2 * 2 * z / (x**2 + y**2 + z**2 + params["d"]**2)
    return (f, df_dx, df_dy, df_dz)

class LogarithmicPotentialJHB(Potential):

    def __init__(self, v_halo, d, coordinate_system=CartesianCoordinates, \
                 length_unit=u.kpc, time_unit=u.Myr, mass_unit=u.solMass):
        ''' Represents a triaxial Logarithmic potential (e.g. triaxial halo).

            $\Phi_{halo} = v_{halo}^2\ln(x^2 + y^2/p^2 + z^2/q^2 + c^2)$

        '''
        super(LogarithmicPotentialJHB, self).\
            __init__(coordinate_system=coordinate_system)
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

        if coordinate_system == CartesianCoordinates:
            f_derivs = _logarithmic_model_cartesian_jhb(self.params)
        else:
            raise ValueError("'coordinate_system' can only be cartesian.")

        self.coordinate_system = coordinate_system
        self.add_component(uuid.uuid4(), f_derivs[0], derivs=f_derivs[1:])
        self.add_component = lambda *args: \
            _raise(AttributeError("'add_component' is only available for the "
                                  "base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{halo} = v_{halo}^2\\ln(x^2 + y^2 + z^2 + d^2)$"

"""