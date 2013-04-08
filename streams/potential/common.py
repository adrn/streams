# coding: utf-8

""" Common Galactic potential components. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import uuid

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G

from .core import CartesianPotential

#import _common # minimal gains from using Cython

__all__ = ["PointMassPotential", "MiyamotoNagaiPotential", "HernquistPotential",
           "LogarithmicPotentialLJ"]

def _raise(ex):
    raise ex

####################################################################################
#    Potential due to a point mass at the given position
####################################################################################

def _cartesian_point_mass_model(G):
    ''' A function that accepts model parameters, and returns a tuple of
        functions that accept coordinates and evaluates this component of the
        potential and its derivatives for cartesian coordinates.
        
        params[0] = mass
        params[1] = origin
            origin[0] = x0, etc.
    '''
    
    def f(x,y,z,m,origin): 
        return -G * m / np.sqrt((x-origin[0])**2 + (y-origin[1])**2 + (z-origin[2])**2)
    
    def df(x,y,z,m,origin): 
        denom = ((x-origin[0])**2 + (y-origin[1])**2 + (z-origin[2])**2)**1.5
        
        dx = G * m*(x-origin[0]) / denom
        dy = G * m*(y-origin[1]) / denom
        dz = G * m*(z-origin[2]) / denom
        
        return np.array([dx,dy,dz])
        
    return (f, df)

class PointMassPotential(CartesianPotential):

    def __init__(self, unit_bases, m, origin=[0.,0.,0.]*u.kpc):
        ''' Represents a point-mass potential at the given origin.

            $\Phi = -\frac{GM}{x-x_0}$

            Parameters
            ----------
            M : astropy.units.Quantity
                Mass.
            origin : astropy.units.Quantity
                Must specify the location of the point mass along each 
                dimension. For example, it could look like 
                    origin=[0,0,0)]*u.kpc

        '''
        super(PointMassPotential, self).\
            __init__(unit_bases)

        # First see if M is a Quantity-like object
        if not isinstance(m, u.Quantity):
            raise TypeError("mass must be an Astropy Quantity object. You "
                            "passed a {0}.".format(type(m)))
        
        if not isinstance(origin, u.Quantity):
            raise TypeError("origin must be an Astropy Quantity object. You "
                            "passed a {0}.".format(type(origin)))
        
        name = uuid.uuid4()
        params = dict(m=m.decompose(bases=self.unit_bases).value, 
                      origin=origin.decompose(bases=self.unit_bases).value)

        f,df = _cartesian_point_mass_model(G.decompose(bases=unit_bases).value)
        self.add_component(name, f, f_prime=df, parameters=params)
        
        # now that we've used it, throw it away
        self.add_component = lambda *args,**kwargs: \
            _raise(AttributeError("'add_component' is only available for the "
                                  "base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi = -\\frac{GM}{r-r_0}$"

####################################################################################
#    Miyamoto-Nagai Disk potential from Miyamoto & Nagai 1975
#    http://adsabs.harvard.edu/abs/1975PASJ...27..533M
####################################################################################

def _cartesian_miyamoto_nagai_model(G):
    ''' A function that accepts model parameters, and returns a tuple of
        functions that accept coordinates and evaluates this component of the
        potential and its derivatives for cartesian coordinates.
    '''
    
    def f(x,y,z,m,a,b,origin): 
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        return -G * m / np.sqrt(xx**2 + yy**2 + (a + np.sqrt(zz**2 + b**2))**2)
    
    def df(x,y,z,m,a,b,origin): 
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        
        denom = ((xx**2 + yy**2) + (a + np.sqrt(zz**2 + b**2))**2)**1.5
        
        dx = G*m*xx / denom
        dy = G*m*yy / denom
        _tmp = a/(np.sqrt(zz**2 + b**2))
        dz = G*m*zz * (1.+_tmp) / denom
                
        return np.array([dx,dy,dz])
        
    return (f, df)

class MiyamotoNagaiPotential(CartesianPotential):

    def __init__(self, unit_bases, m, a, b, origin=[0.,0.,0.]*u.kpc):
        ''' Represents the Miyamoto-Nagai potential (1975) for a disk-like
            potential.

            $\Phi_{disk} = -\frac{GM_{disk}}{\sqrt{R^2 + (a + sqrt{z^2 + b^2})^2}}$

        '''
        super(MiyamotoNagaiPotential, self).\
            __init__(unit_bases)

        # First see if parameters is a Quantity-like object
        if not isinstance(m, u.Quantity) or not isinstance(a, u.Quantity) \
            or not isinstance(b, u.Quantity) or not isinstance(origin, u.Quantity):
            raise TypeError("parameters must be Astropy Quantity objects.")
        
        name = uuid.uuid4()
        params = dict(m=m.decompose(bases=unit_bases).value, 
                      origin=origin.decompose(bases=unit_bases).value, 
                      a=a.decompose(bases=unit_bases).value, 
                      b=b.decompose(bases=unit_bases).value)

        f,df = _cartesian_miyamoto_nagai_model(G.decompose(bases=unit_bases).value)
        self.add_component(name, f, f_prime=df, parameters=params)
        
        # now that we've used it, throw it away
        self.add_component = lambda *args,**kwargs: \
            _raise(AttributeError("'add_component' is only available for the "
                                  "base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{disk} = -\\frac{GM_{disk}}{\\sqrt{R^2 + (a + \\sqrt{z^2 + b^2})^2}}$"

####################################################################################
#    Hernquist Spheroid potential from Hernquist 1990
#    http://adsabs.harvard.edu/abs/1990ApJ...356..359H
####################################################################################

def _cartesian_hernquist_model(G):
    ''' A function that accepts model parameters, and returns a tuple of
        functions that accept coordinates and evaluates this component of the
        potential and its derivatives for cartesian coordinates.
    '''
    
    def f(x,y,z,m,c,origin): 
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        val = -G * m / (np.sqrt(xx**2 + yy**2 + zz**2) + c)
        return val
    
    def df(x,y,z,m,c,origin):
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        
        denom = (np.sqrt(xx**2 + yy**2 + zz**2) + c)**2 * np.sqrt(xx**2 + yy**2 + zz**2)
        
        dx = G*m*xx / denom
        dy = G*m*yy / denom
        dz = G*m*zz / denom
        
        return np.array([dx,dy,dz])
        
    return (f, df)
    
class HernquistPotential(CartesianPotential):
    
    def __init__(self, unit_bases, m, c, origin=[0.,0.,0.]*u.kpc):
        ''' Represents the Hernquist potential (1990) for a spheroid (bulge).

            $\Phi_{spher} = -\frac{GM_{spher}}{r + c}$
        '''
        super(HernquistPotential, self).\
            __init__(unit_bases)

        # First see if parameters is a Quantity-like object
        if not isinstance(m, u.Quantity) or not isinstance(c, u.Quantity) \
            or not isinstance(origin, u.Quantity):
            raise TypeError("parameters must be Astropy Quantity objects.")
        
        name = uuid.uuid4()
        params = dict(m=m.decompose(bases=unit_bases).value, 
                      origin=origin.decompose(bases=unit_bases).value, 
                      c=c.decompose(bases=unit_bases).value)

        f,df = _cartesian_hernquist_model(G.decompose(bases=unit_bases).value)
        self.add_component(name, f, f_prime=df, parameters=params)
        
        # now that we've used it, throw it away
        self.add_component = lambda *args,**kwargs: \
            _raise(AttributeError("'add_component' is only available for the "
                                  "base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{spher} = -\\frac{GM_{spher}}{r + c}$"

####################################################################################
#    Triaxial, Logarithmic potential (see: Johnston et al. 1998)
#    http://adsabs.harvard.edu/abs/1999ApJ...512L.109J
####################################################################################

def _cartesian_logarithmic_lj_model(G):
    ''' A function that accepts model parameters, and returns a tuple of
        functions that accept coordinates and evaluates this component of the
        potential and its derivatives for cartesian coordinates.

        Potential from David Law's paper?
    '''

    def f(x,y,z,v_halo,q1,q2,qz,phi,r_halo,origin): 
        C1 = (np.cos(phi)/q1)**2+(np.sin(phi)/q2)**2
        C2 = (np.cos(phi)/q2)**2+(np.sin(phi)/q1)**2
        C3 = 2.*np.sin(phi)*np.cos(phi)*(1./q1**2 - 1./q2**2)
        
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        
        return v_halo**2 * np.log(C1*xx**2 + C2*yy**2 + C3*xx*yy + zz**2/qz**2 + r_halo**2)
    
    def df(x,y,z,v_halo,q1,q2,qz,phi,r_halo,origin):
        C1 = (np.cos(phi)/q1)**2+(np.sin(phi)/q2)**2
        C2 = (np.cos(phi)/q2)**2+(np.sin(phi)/q1)**2
        C3 = 2.*np.sin(phi)*np.cos(phi)*(1./q1**2 - 1./q2**2)
        
        xx = (x-origin[0])
        yy = (y-origin[1])
        zz = (z-origin[2])
        
        denom = (C1*xx**2 + C2*yy**2 + C3*xx*yy + zz**2/qz**2 + r_halo**2)
        
        dx = v_halo**2 * (2.*C1*xx + C3*yy) / denom
        dy = v_halo**2 * (2.*C2*yy + C3*xx) / denom
        dz = 2.*v_halo**2 * zz / denom / qz**2
        
        return np.array([dx,dy,dz])
        
    return (f, df)

class LogarithmicPotentialLJ(CartesianPotential):

    def __init__(self, unit_bases, **kwargs):
        ''' Represents a triaxial Logarithmic potential (e.g. triaxial halo).
            
            $\Phi_{halo} = v_{halo}^2\ln(C1x^2 + C2y^2 + C3xy + z^2/q_z^2 + r_halo^2)$
            
            Parameters: v_halo, q1, q2, qz, phi, r_halo, origin

        '''
        super(LogarithmicPotentialLJ, self).__init__(unit_bases)
        
        # First see if parameters is a Quantity-like object
        assert "q1" in kwargs.keys() and \
               "q2" in kwargs.keys() and \
               "qz" in kwargs.keys()
        
        if "origin" not in kwargs.keys():
            kwargs["origin"] = [0.,0.,0.]*u.kpc
        
        for k in ["v_halo", "phi", "r_halo", "origin"]:
            if not isinstance(kwargs[k], u.Quantity):
                raise TypeError("parameters must be Astropy Quantity objects.")
        
        name = uuid.uuid4()
        params = dict()
        for key,val in kwargs.items():
            if isinstance(val, u.Quantity):
                params[key] = val.decompose(bases=unit_bases).value
            else:
                params[key] = val

        f,df = _cartesian_logarithmic_lj_model(G.decompose(bases=unit_bases).value)
        self.add_component(name, f, f_prime=df, parameters=params)
        
        # now that we've used it, throw it away
        self.add_component = lambda *args,**kwargs: \
            _raise(AttributeError("'add_component' is only available for the "
                                  "base Potential class."))

    def _repr_latex_(self):
        ''' Custom latex representation for IPython Notebook '''
        return "$\\Phi_{halo} = v_{halo}^2\\ln(C1x^2 + C2y^2 + C3xy + z^2/q_z^2 + r_halo^2)$"

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