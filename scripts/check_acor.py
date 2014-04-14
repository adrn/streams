# coding: utf-8

""" Check autocorrelation times """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
from emcee import autocorr
import numpy as np
import h5py

# Project
from streams.inference import StreamModel
import streams.io as io

def main(config_file):
    # read configuration from a YAML file
    config = io.read_config(config_file)
    output_path = config["output_path"]
    cache_path = os.path.join(output_path,"cache")

    for filename in glob.glob(os.path.join(cache_path,"inference_1*.hdf5")):
        with h5py.File(filename, "r") as f:
            try:
                chain = np.hstack((chain,f["chain"].value))
            except NameError:
                chain = f["chain"].value
            accfr = f["acceptance_fraction"].value

            _a = (np.min(accfr),np.median(accfr),np.max(accfr))
            print("min, median, max: {}, {}, {}".format(_a))

    acf = autocorr.function(np.mean(chain, axis=0), axis=0)

    plt.clf()
    for ii in range(4):
        plt.plot(acf[:,ii], marker=None, alpha=0.75)
    plt.savefig(os.path.join(output_path, "acf.png"))

    windows = np.logspace(1, 4, 25)
    acors = []
    for window in windows:
        _acor = autocorr.integrated_time(np.mean(chain, axis=0),
                                         axis=0, window=window) # 50 comes from emcee
        acors.append(np.median(_acor))

    plt.clf()
    v = plt.loglog(windows, acors, linestyle='none', marker='o')
    plt.ylim(1,1e4)
    plt.savefig(os.path.join(output_path, "acor_windows.png"))

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-f", "--file", dest="file", required=True,
                        help="Path to the configuration file to run with.")
    args = parser.parse_args()

    main(args.file)