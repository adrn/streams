# coding: utf-8

""" Select targets for Spitzer """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from datetime import datetime, timedelta

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column, join
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.util import project_root

def orphan():
    """ We'll select the high probability members from Branimir's
        Orphan sample.
    """
    filename = "branimir_orphan.txt"
    output_file = "orphan.txt"

    d = ascii.read(os.path.join(project_root, "data", "catalog", filename))
    high = d[d["membership_probability"] == "high"]
    high.keep_columns(["ID", "RA", "Dec", "magAvg", "period", "rhjd0"])
    high.rename_column("magAvg", "rMagAvg")

    ascii.write(high, \
        os.path.join(project_root, "data", "spitzer_targets", output_file))

if __name__ == "__main__":
    orphan()