# coding: utf-8

""" TriAnd RR Lyrae """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from datetime import datetime, timedelta

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.observation.rrlyrae import extrapolate_light_curve, time_to_phase
from streams.util import project_root

data_file = os.path.join(project_root, "data", "catalog", "TriAnd_RRLyr.txt")
stars = ascii.read(data_file, 
                   converters={'objectID' : [ascii.convert_numpy(np.str)]},
                   header_start=0,
                   data_start=1,
                   delimiter=" ")

kitt_peak_longitude = (111. + 35/60. + 40.9/3600)*u.deg

def gmst_time_to_utc(t):
    jd = int(t.jd) + 0.5
    
    S = jd - 2451545.0
    T = S / 36525.0
    T0 = 6.697374558 + (2400.051336*T) + (0.000025862*T**2)
    T0 = T0 % 24
    
    h = (t.jd - jd)*24.
    GST = (h - T0) % 24
    UT = GST * 0.9972695663
    
    tt = Time(jd, format='jd', scale='utc')
    dt = tt.datetime + timedelta(hours=UT)
    
    return Time(dt, scale='utc')

def source_meridian_window(ra, day, buffer_time=2.*u.hour):
    """ Compute the minimum and maximum time (UTC) for the window 
        of observability for the given source.
        
        Parameters
        ----------
        ra : astropy.units.Quantity
            The right ascension of the object.
        day : astropy.time.Time
            The day to compute the window on.
    """
    
    hour_angles = [-buffer_time.hour,buffer_time.hour]*u.hourangle
    
    jds = []
    for ha in hour_angles:
        lst = day.datetime + timedelta(hours=(ha + ra).hourangle)
        gmst = lst + timedelta(hours=kitt_peak_longitude.hourangle)
        jds.append(gmst_time_to_utc(Time(gmst, scale='utc')).jd)
    
    return Time(jds, scale='utc', format='jd')

# stuff
day = Time(datetime(2013, 8, 26), scale='utc')
for star in stars:
    t1,t2 = source_meridian_window(star['ra']*u.deg, day)
    if t1.jd > day.jd+1.:
        times = Time(np.linspace(t1.jd-1, t2.jd-1, 100), format='jd', scale='utc')
    else:
        times = Time(np.linspace(t1.jd, t2.jd, 100), format='jd', scale='utc')
    
    period = star['period']*u.day
    t0 = Time(star['rhjd0'], scale='utc', format='mjd')
    phases = time_to_phase(times, period=period, t0=t0)
    print(phases)
    sys.exit(0)
    mag = extrapolate_light_curve(times, period=period, t0=t0)
    
    plt.plot(times.mjd, mag)

plt.ticklabel_format(axis='x', style='plain', useOffset=False)
plt.show()