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
    parser.add_argument("--date", dest="ut_date", default=None,
                        type=str, help="e.g., 2013-10-23")
    parser.add_argument("--time", dest="ut_time", default=None,
                        type=str, help="e.g., 01:53")

    args = parser.parse_args()

    if args.ut_time:
        _time = time(*map(int, map(float, args.ut_time.split(":"))))
    else:
        _time = datetime.utcnow().time()

    if args.ut_date:
        _date = date(*map(int, map(float, args.ut_date.split("-"))))
    else:
        _date = datetime.utcnow().date()

    tt = datetime.combine(_date, _time)
    t = Time(tt, scale='utc')

    if t is None:
        raise ValueError("Must specify now or utc")

    object_name = args.obj_name.strip().replace(" ", "_")

    this_star = filter(lambda s: s['name'] == object_name, all_stars)[0]
    period = this_star['period']*u.day
    t0 = Time(this_star['rhjd0'], scale='utc', format='mjd')

    step = 20.*u.minute
    jds = Time(np.arange(t.jd, t.jd+(4/72.), step.day),
               format='jd', scale='utc')

    day_str = "UTC Day: {0}".format(_date)
    print("-"*len(day_str))
    print(day_str)
    print("-"*len(day_str))
    print("Time Phase")
    for jd in jds:
        print(jd.datetime.time().strftime("%H:%M"), \
              "{0:.2f}".format(time_to_phase(jd, period=period, t0=t0)))
    #phases = time_to_phase(jds, period=period, t0=t0)
