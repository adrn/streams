# coding: utf-8

""" Test the MPI pool """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import astropy.units as u
from emcee.utils import MPIPool
import numpy as np

def test_function(arr):
    return 0.

def main():
    Npool = 4
    x = [np.random.random(25,10,6) for ii in range(Npool)]

    pool = MPIPool()
    pool.map(test_function, )

if __name__ == "__main__":
    main()