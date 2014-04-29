# coding: utf-8

""" Given a config file, create a master inference.hdf5 file. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import glob

# Third-party
import acor
import h5py
import numpy as np
import astropy.units as u

# Project
from streams import usys
from streams.io import read_hdf5, read_config
from streams.inference import StreamModel
from streams.util import streamspath

def master_inference(filename):
    """ Create a master inference hdf5 file from output every 1000 steps """

    cache_path = os.path.join(streamspath, "plots", "infer_potential",
                              filename, "cache")

    for filename in sorted(glob.glob(os.path.join(cache_path,"inference_*.hdf5"))):
        print(filename)
        with h5py.File(filename, "r") as f:
            try:
                chain = np.hstack((chain,f["chain"].value))
            except NameError:
                chain = f["chain"].value

            accfrac = f["acceptance_fraction"].value

    #taur = [acor.acor(chain[:,:,i])[0] for i in range(chain.shape[2])]
    tau,mm,xx = acor.acor(np.mean(chain[accfrac > 0.02],axis=0).T)
    acor_time = int(2*np.max(tau))
    print("Autocorrelation time: ", acor_time)

    fn = os.path.join(cache_path, "combined_inference.hdf5")

    with h5py.File(fn, "w") as f:
        f["chain"] = chain[:,::acor_time].copy()

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-f", "--file", dest="filename", required=True,
                        help="Path to the configuration file to run with.")

    args = parser.parse_args()

    master_inference(args.filename)
