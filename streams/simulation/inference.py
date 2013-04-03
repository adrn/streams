# coding: utf-8

""" Contains priors and likelihood functions for inferring parameters of
    the Logarithmic potential using back integration.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

from streams.potential.lm10 import param_ranges

def ln_p_qz(qz):
    """ Prior on vertical (z) axis ratio """
    lo,hi = param_ranges["qz"]
    
    if qz <= lo or qz >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_q1(q1):
    """ Prior on axis ratio """
    lo,hi = param_ranges["q1"]
    
    if q1 <= lo or q1 >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_q2(q2):
    """ Prior on axis ratio """
    lo,hi = param_ranges["q2"]
    
    if q2 <= lo or q2 >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_v_halo(v):
    """ Prior on mass of the halo (v_halo). The range imposed is roughly a
        halo mass between 10^10 and 10^12 M_sun at 200 kpc
    """
    lo,hi = param_ranges["v_halo"]
    
    if v <= lo or v >= hi:
        return -np.inf
    else:
        return 0.

def ln_p_phi(phi):
    """ Prior on orientation angle between DM halo and disk """
    lo,hi = param_ranges["phi"]
    
    if phi < lo or phi > hi:
        return -np.inf
    else:
        return 0.

def ln_p_r_halo(r_halo):
    """ Prior on halo concentration parameter """
    lo,hi = param_ranges["r_halo"]
    
    if r_halo < lo or r_halo > hi:
        return -np.inf
    else:
        return 0.

def ln_prior(p):
        sum = 0
        for ii in range(len(p)):
            sum += prior_map[param_map[ii]](p[ii])
        return sum

_prior_map = dict(qz=ln_p_qz, q1=ln_p_q1, q2=ln_p_q2, v_halo=ln_p_v_halo, \
                        phi=ln_p_phi, r_halo=ln_p_r_halo)

def make_prior(param_names):
    """ Return a *function* that evaluates the prior for a set of parameters """
    
    # Construct the prior based on the requested parameters
    _param_map = dict(zip(range(len(param_names)), param_names))
    
    def ln_prior(p):
        sum = 0
        for ii,param_name in enumerate(param_names):
            sum += _prior_map[param_name](p[ii])
        return sum
    
    return ln_prior