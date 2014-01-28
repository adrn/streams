# coding: utf-8

""" Read in the VL2 stream data, output a JSON file for Three.js """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import json
import logging

# Third-party
import numpy as np

# project
from streams.util import project_root

# Create logger
logger = logging.getLogger(__name__)

def main(N=None):
    path = os.path.join(project_root, "data", "VL2 Streams")
    all_filenames = glob.glob(os.path.join(path, "*.dat"))

    xyz = dict()
    for ii,filename in enumerate(all_filenames):
        with open(filename) as f:
            d = np.loadtxt(f)
            if N is not None:
                xyz[str(ii)] = dict(data=d[1:N+1,:3].tolist())
            else:
                xyz[str(ii)] = dict(data=d[1:,:3].tolist())

    output_file = os.path.join(path,
                    "{}_vl2.json".format(N if N is not None else 'all'))
    with open(output_file, 'w') as f:
        f.write(json.dumps(xyz))

if __name__ == "__main__":
    main(N=5000)