# coding: utf-8

""" TriAnd RR Lyrae """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from datetime import datetime, timedelta, time, date

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Column
from astropy.time import Time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.coordinates import sex_to_dec
from streams.observation.time import gmst_to_utc, lmst_to_gmst
from streams.observation.rrlyrae import time_to_phase, phase_to_time
from streams.observation.triand import all_stars, triand_stars, standards
from streams.util import project_root

# CHANGE THIS
kitt_peak_longitude = (111. + 35/60. + 40.9/3600)*u.deg

def main():

    tbl = ascii.read("/Users/adrian/Downloads/jdlist.txt", delimiter=",",
                     names=["file_name", "object_name", "jd"])
    phase_data = np.zeros(len(tbl))*np.nan
    for i,line in enumerate(tbl):
        t = Time(line['jd'], format='jd', scale='utc')
        try:
            object_name = line['object_name'].strip().replace(" ", "_")
            this_star = filter(lambda s: s['name'] == object_name, all_stars)[0]
        except IndexError:
            print("Skipping {}...".format(object_name))
            continue
        except AttributeError:
            print("No name in this line!")
            continue

        period = this_star['period']*u.day
        t0 = Time(this_star['rhjd0'], scale='utc', format='mjd')
        phase = time_to_phase(t, period=period, t0=t0)
        phase_data[i] = phase

    col = Column(phase_data,"phase")
    tbl.add_column(col)
    ascii.write(tbl, "/Users/adrian/Downloads/jdlist-with-phase.txt", delimiter=",")

if __name__ == "__main__":
    main()