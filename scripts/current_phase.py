# coding: utf-8

""" TriAnd RR Lyrae """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from datetime import datetime, timedelta, time

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.time import Time
from astropy.table import Table, Column, join
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.coordinates import sex_to_dec
from streams.observation.time import gmst_to_utc, lmst_to_gmst
from streams.observation.rrlyrae import time_to_phase, phase_to_time
from streams.observation.triand import all_stars, triand_stars, standards
from streams.util import project_root

matplotlib.rc('xtick', labelsize=12, direction='in')
matplotlib.rc('ytick', labelsize=12, direction='in')

# CHANGE THIS
output_path = "/Users/adrian/Documents/GraduateSchool/Observing/2013-10_MDM"
kitt_peak_longitude = (111. + 35/60. + 40.9/3600)*u.deg

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("--object", dest="obj_name", default=None,
                        required=True, type=str, help="e.g., RR Lyr")
    parser.add_argument("--utc", dest="ut_time", default=None,
                        type=str, help="e.g., 01:53")
    parser.add_argument("--now", dest="now", action="store_true",
                        default=False, help="Use current time")

    args = parser.parse_args()

    if args.now:
        t = Time(datetime.utcnow(), scale='utc')
    elif args.ut_time:
        tup = map(int, map(float, args.ut_time.split(":")))
        #tt = datetime.combine(datetime.utcnow().date(), time(*tup))
        tt = datetime.combine(datetime(2013, 10, 23), time(*tup))
        t = Time(tt, scale='utc')
    else:
        raise ValueError("Must specify now or utc")

    object_name = args.obj_name.strip().replace(" ", "_")

    this_star = filter(lambda s: s['name'] == object_name, all_stars)[0]
    period = this_star['period']*u.day
    t0 = Time(this_star['rhjd0'], scale='utc', format='mjd')

    step = 20.*u.minute
    jds = Time(np.arange(t.jd, t.jd+(4/72.), step.day),
               format='jd', scale='utc')

    print("Time Phase")
    for jd in jds:
        print(jd.datetime.time().strftime("%H:%M"), \
              "{0:.2f}".format(time_to_phase(jd, period=period, t0=t0)))
    #phases = time_to_phase(jds, period=period, t0=t0)
