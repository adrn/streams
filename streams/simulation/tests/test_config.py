# coding: utf-8
"""
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cStringIO as StringIO

# Third-party
import numpy as np
import pytest
import astropy.units as u

from ..config import read

def test_read():

    file = """(I) particles : 100 # number of particles
              (U) dt : 1. Myr # timestep for back integration
              (B) with_errors : yes
              (S) description : blah blah blah
              (L,S) parameters : q1 qz v_halo phi
              (L,S) expr : tub>0 (x**2+y**2+z**2)<100"""
    
    f = StringIO.StringIO(file)
    config = read(f)
    
    assert config["particles"] == 100
    assert config["dt"] == (1.*u.Myr)
    assert config["with_errors"] == True
    assert config["parameters"] == ["q1", "qz", "v_halo", "phi"]
    assert config["expr"] == ["tub>0", "(x**2+y**2+z**2)<100"]

def test_read_multiline():
    
    file = """(M,I) particles : 100 # number of particles
              (M,S) expr : (Pcol>0) & (abs(Lmflag)==1)
              (M,I) particles : 50
              (M,S) expr : (Pcol>0) & (abs(Lmflag)==2)
              (U) dt : 1. Myr # timestep for back integration
              (B) with_errors : yes
              (S) description : blah blah blah
              (L,S) parameters : q1 qz v_halo phi
           """
    
    f = StringIO.StringIO(file)
    config = read(f)
    
    assert config["dt"] == (1.*u.Myr)
    assert config["with_errors"] == True
    assert config["parameters"] == ["q1", "qz", "v_halo", "phi"]
    assert config["expr"] == ["(Pcol>0) & (abs(Lmflag)==1)", "(Pcol>0) & (abs(Lmflag)==2)"]
    assert config["particles"] == [100, 50]