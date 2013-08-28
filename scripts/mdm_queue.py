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
from astropy.table import Table, Column, join
import matplotlib.pyplot as plt
import numpy as np

# Project
from streams.observation.rrlyrae import time_to_phase, phase_to_time
from streams.util import project_root

data_file = os.path.join(project_root, "data", "catalog", "TriAnd_RRLyr.txt")
stars = ascii.read(data_file, 
                   converters={'objectID' : [ascii.convert_numpy(np.str)]},
                   header_start=0,
                   data_start=1,
                   delimiter=" ")
ras = np.array(stars['ra'])
ras[ras > 90.] = ras[ras > 90.] - 360.
idx = np.argsort(ras)
stars = stars[idx]

names = ["TriAndRRL{0}".format(ii+1) for ii in range(len(stars))]
stars.add_column(Column(names, name='name'))

# Read in RR Lyrae standards
RRLyr_stds1 = ascii.read("/Users/adrian/Documents/GraduateSchool/Observing Runs/Std RR Lyrae/nemec_RRLyrae.txt")
RRLyr_stds2 = ascii.read("/Users/adrian/Documents/GraduateSchool/Observing Runs/Std RR Lyrae/bi-qing_for2011_RRLyr.txt", delimiter=',')
RRLyr_stds2.add_column(Column(RRLyr_stds2['hjd0_2450000'] + 2450000., name='rhjd0'))
RRLyr_stds2['ra'] = [coord.Angle(x, unit=u.hour).degree for x in RRLyr_stds2['ra_str']]
RRLyr_stds2['dec'] = [coord.Angle(x, unit=u.degree).degree for x in RRLyr_stds2['dec_str']]

#stars = join(stars, RRLyr_stds1, join_type='outer')
stars = join(stars, RRLyr_stds2, join_type='outer')

kitt_peak_longitude = (111. + 35/60. + 40.9/3600)*u.deg

def print_tcs_list(table):
    """ Given the table of stars, ouput a list to be fed in to the TCS """
    
    for ii,star in enumerate(stars):
        name = "TriAndRRL{0}".format(ii+1)
        ra = coord.RA(star['ra']*u.deg)
        dec = coord.Dec(star['dec']*u.deg)
        print("{name}\t{ra}\t{dec}\t2000.0".format(name=name,
                                                ra=ra.to_string(unit=u.hour, sep=' ', pad=True, precision=1),
                                                dec=dec.to_string(unit=u.deg, sep=' ', alwayssign=True, precision=1)))

def print_tcs_list_decimal(table):
    """ Given the table of stars, ouput a list to be fed in to the TCS """
    
    for ii,star in enumerate(stars):
        name = "TriAndRRL{0}".format(ii+1)
        print("{name}\t{ra}\t{dec}\t2000.0".format(name=name,
                                                ra=star['ra'],
                                                dec=star['dec']))

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

def open_finder_charts():
    for ii,star in enumerate(stars):
        url = "http://ptf.caltech.edu/cgi-bin/ptf/variable/fc.cgi?name=TriAndRRL{0}&ra={ra}&dec={dec}&bigwidth=300.000000&bigspan=19.000000&zoomwidth=150.000000&zoomspan=8.000000"
        print(url.format(ii+1, ra=star['ra'], dec=star['dec']))
        os.system("open '{0}'".format(url.format(ii+1, ra=star['ra'], dec=star['dec'])))

#print_tcs_list(stars)

queue = []

# stuff
day = Time(datetime(2013, 8, 29), scale='utc')
for star in stars:
    # For each star, figure out its observability window, e.g., the times
    #   that it is at -2 hr from meridian and +2 hr from meridian
    t1,t2 = source_meridian_window(star['ra']*u.deg, day)
    if t1.jd > day.jd+1.:
        jd1 = t1.jd-1
        jd2 = t2.jd-1
    else:
        jd1 = t1.jd
        jd2 = t2.jd
    
    obs_window = Time([jd1, jd2], format='jd', scale='utc')
    period = star['period']*u.day
    t0 = Time(star['rhjd0'], scale='utc', format='mjd')
    
    # Now we want to see if at what time the phase of the pulsation is around
    #   0.4 and 0.7
    #optimal_phases = np.array([0.4, 0.7])
    #optimal_phase_times = phase_to_time(optimal_phases, day=day, t0=t0, period=period)
    #print(optimal_phase_times.datetime, obs_window.datetime)
    #continue
    
    # Here instead we'll look at the possible times we can observe
    step = 30.*u.minute
    jds = Time(np.arange(jd1, jd2+step.day, step.day), format='jd', scale='utc')
    ut_hours = np.array([jd.datetime.hour for jd in jds])
    phases = time_to_phase(jds, period=period, t0=t0)
    idx1 = (phases > 0.1) & (phases < 0.5) & (ut_hours < 13.) & (ut_hours > 3.)
    idx2 = (phases >= 0.5) & (phases < 0.7) & (ut_hours < 13.) & (ut_hours > 3.)
    
    if not np.any(idx1) and not np.any(idx2):
        print("Skipping {0}, no suitable observing time.".format(star['name']))
        continue
    
    if np.any(idx1):
        first_obs_time = jds[idx1][0]
        phase_at_first = phases[idx1][0]
        queue.append({'name' : star['name'], 
                      'time' : first_obs_time.datetime,
                      'phase' : phase_at_first,
                      'info' : "pre-mean"})
    
    if np.any(idx2):
        first_obs_time = jds[idx2][0]
        phase_at_first = phases[idx2][0]
        queue.append({'name' : star['name'], 
                      'time' : first_obs_time.datetime,
                      'phase' : phase_at_first,
                      'info' : "post-mean"})

queue = Table(queue)
queue = queue['name','time','phase','info']
queue.sort('time')
ascii.write(queue, '/Users/adrian/Downloads/test.txt', Writer=ascii.Basic)
