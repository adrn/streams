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
    w = arr * 10.
    return 0.

def main():

    v = [np.random.random(size=(100000,)) for ii in range(100)]

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    y = pool.map(test_function, v)
    pool.close()

if __name__ == "__main__":
    main()
