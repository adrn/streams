# coding: utf-8

""" Make a table with object name, date, exposure start time, phase """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
from datetime import datetime, timedelta
from collections import defaultdict

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii, fits
from astropy.time import Time
from astropy.table import Table, Column, join
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.coordinates import sex_to_dec
from streams.observation.time import gmst_to_utc, lmst_to_gmst
from streams.observation.rrlyrae import time_to_phase, phase_to_time
from streams.observation.triand import all_stars
from streams.util import project_root

def run_stats(data_path):
    tbl_dict = defaultdict(list)
    for m_path in glob.glob(os.path.join(data_path, "m*")):
        for fname in glob.glob(os.path.join(m_path, "*.fit")):
            hdr = fits.getheader(fname)

            if hdr["IMAGETYP"] != "OBJECT":
                continue

            object_name = hdr["OBJECT"].strip()
            ut_date = hdr["DATE-OBS"]
            ut_time = hdr["TIME-OBS"]

            obj_und = object_name.replace(" ", "_")
            try:
                star = filter(lambda r: r["name"] == obj_und, all_stars)[0]
            except IndexError:
                print("Failed for: {0}".format(object_name))
                continue

            period = star['period']*u.day
            t0 = Time(star['rhjd0'], scale='utc', format='mjd')

            date_time = ut_date.split("-") + ut_time.split(":")
            dt = datetime(*map(int, map(float, date_time)))
            t = Time(dt, scale='utc')
            phase = time_to_phase(t, period=period, t0=t0)

            tbl_dict["name"].append(object_name)
            tbl_dict["date"].append(ut_date)
            tbl_dict["start time"].append(ut_time)
            tbl_dict["phase"].append(phase)

    t = Table(tbl_dict)
    return t

def main(overwrite=False):
    observing_path = "/Users/adrian/Documents/GraduateSchool/Observing/"
    run_path = os.path.join(observing_path, "2013-10_MDM")
    data_path = os.path.join(run_path, "data")

    cache_name = os.path.join(data_path, "stats.tbl")

    if os.path.exists(cache_name) and overwrite:
        os.remove(cache_name)

    if not os.path.exists(cache_name):
        t = run_stats(data_path)
        ascii.write(t, cache_name, Writer=ascii.Basic)

    t = ascii.read(cache_name)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    for name in np.unique(t["name"]):
        star_data = t[t["name"] == name]

        ax.cla()
        ax.plot(star_data["phase"], star_data["phase"],
                marker='o', linestyle='none', alpha=0.75)
        ax.set_title(name)
        ax.set_xlabel("Phase")
        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        fig.savefig(os.path.join(run_path, "phase_{0}.png".format(name)))

if __name__ == "__main__":
    main()
