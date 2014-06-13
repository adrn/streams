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

def master_inference(path, all=False, outfile="combined_inference.hdf5"):
    """ Create a master inference hdf5 file from output every 1000 steps """

    # first see if relative
    cache_path = os.path.join(streamspath, path)
    print(cache_path)
    if not os.path.exists(cache_path):
        raise IOError("Path doesn't exist!")

    for filename in sorted(glob.glob(os.path.join(cache_path,"inference_*.hdf5"))):
        print(filename)
        with h5py.File(filename, "r") as f:
            try:
                chain = np.hstack((chain,f["chain"].value))
            except NameError:
                chain = f["chain"].value

            accfrac = f["acceptance_fraction"].value

    if not all:
        #taur = [acor.acor(chain[:,:,i])[0] for i in range(chain.shape[2])]
        tau,mm,xx = acor.acor(np.mean(chain[accfrac > 0.02],axis=0).T)
        acor_time = int(2*np.max(tau))
        print("Autocorrelation times: ", tau)
        print("Max autocorrelation time: ", acor_time)

        _chain = chain[:,::acor_time].copy()

    else:
        _chain = chain.copy()

    fn = os.path.join(cache_path, outfile)

    with h5py.File(fn, "w") as f:
        f["chain"] = _chain
        f["acor"] = tau

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("--path", dest="path", required=True,
                        help="Path to the inference files relative to $STREAMSPATH.")
    parser.add_argument("--all", dest="all", action="store_true",
                        help="Return all steps instead of every max(t_acor).")
    parser.add_argument("--outfile", dest="outfile",
                        help="Name of output file.")

    args = parser.parse_args()
    master_inference(args.path, args.all, args.outfile)
