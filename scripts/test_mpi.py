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
    return w

def main():
    N = 128
    v = np.random.random(size=(N,100000))

    pool = MPIPool()

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    y = pool.map(test_function, v)
    pool.close()

if __name__ == "__main__":
    main()
