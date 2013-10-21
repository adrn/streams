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

def tcs_list(decimal=False):
    """ Given the table of stars, ouput a list to be fed in to the TCS """

    names = []
    ras = []
    decs = []
    epoch = []
    for ii,star in enumerate(triand_stars):
        names.append("TriAndRRL{0}".format(ii+1))

        ra = coord.RA(star['ra']*u.deg)
        dec = coord.Dec(star['dec']*u.deg)

        if decimal:
            ras.append(ra.degree)
            decs.append(dec.degree)
        else:
            ras.append(ra.to_string(unit=u.hour, sep=' ', pad=True, precision=1))
            decs.append(dec.to_string(unit=u.deg, sep=' ', alwayssign=True, precision=1))
        epoch.append(2000.0)

    for ii,star in enumerate(standards):
        names.append(star['name'].replace(' ', '_'))
        ra = coord.RA(star['ra']*u.deg)
        dec = coord.Dec(star['dec']*u.deg)

        if decimal:
            ras.append(ra.degree)
            decs.append(dec.degree)
        else:
            ras.append(ra.to_string(unit=u.hour, sep=' ', pad=True, precision=1))
            decs.append(dec.to_string(unit=u.deg, sep=' ', alwayssign=True, precision=1))
        epoch.append(2000.0)

    t = Table()
    t.add_column(Column(names, name='name'))
    t.add_column(Column(ras, name='ra'))
    t.add_column(Column(decs, name='dec'))
    if not decimal:
        t.add_column(Column(epoch, name='epoch'))

    return t

def source_meridian_window(ra, utc_day, buffer_time=2.*u.hour):
    """ Compute the minimum and maximum time (UTC) for the window
        of observability for the given source.

        Parameters
        ----------
        ra : astropy.units.Quantity
            The right ascension of the object.
        utc_day : astropy.time.Time
            The day to compute the window on.
    """

    hour_angles = [-buffer_time.hour,buffer_time.hour]*u.hourangle

    jds = []
    for ha in hour_angles:
        lst = day.datetime + timedelta(hours=(ha + ra).hourangle)
        gmst = lmst_to_gmst(lst.time(), kitt_peak_longitude)
        utc = gmst_to_utc(gmst, utc_day)
        jds.append(utc.jd)

    return Time(jds, scale='utc', format='jd')

def open_finder_charts():
    for ii,star in enumerate(triand_stars):
        url = "http://ptf.caltech.edu/cgi-bin/ptf/variable/fc.cgi?name=TriAndRRL{0}&ra={ra}&dec={dec}&bigwidth=300.000000&bigspan=19.000000&zoomwidth=150.000000&zoomspan=8.000000"
        print(url.format(ii+1, ra=star['ra'], dec=star['dec']))
        os.system("open '{0}'".format(url.format(ii+1, ra=star['ra'], dec=star['dec'])))

# output list of all targets to be added to the TCS
tcs = tcs_list(False)
fn = os.path.join(output_path, "tcs_list.txt")
ascii.write(tcs, fn, Writer=ascii.Basic)
with open(fn, 'r') as f:
    d = "".join(f.readlines()[1:]).replace('"', '')

with open(fn, 'w') as f:
    f.write(d)

tcs = tcs_list(True)
fn = "iobserve_list.txt"
ascii.write(tcs, os.path.join(output_path, fn), Writer=ascii.Basic)

# Create a queue for the given day
queue = []

_date = map(int, sys.argv[1].split("-"))
day = Time(datetime(*_date), scale='utc')

phase_plot_path = os.path.join(output_path, str(day.datetime.date()))
if not os.path.exists(phase_plot_path):
    os.mkdir(phase_plot_path)

for star in all_stars:
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
    step = 1.*u.minute
    jds = Time(np.arange(jd1, jd2+step.day, step.day),
               format='jd', scale='utc')
    ut_hours = np.array([sex_to_dec((jd.datetime.hour,jd.datetime.minute,jd.datetime.second)) for jd in jds])
    phases = time_to_phase(jds, period=period, t0=t0)

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(ut_hours, phases)
    ax.set_xlim(2., 12.)
    ax.set_ylim(0., 1.)
    ax.text(5., 0.9, star['name'])
    fig.savefig(os.path.join(phase_plot_path, "{0}.png".format(star['name'])))

    idx1 = (phases > 0.1) & (phases < 0.4) & \
           (ut_hours < 13.) & (ut_hours > 2.)
    idx2 = (phases >= 0.5) & (phases < 0.7) & \
           (ut_hours < 13.) & (ut_hours > 2.)

    if not np.any(idx1) and not np.any(idx2):
        print("Skipping {0}, no suitable observing time."\
                .format(star['name']))
        continue

    if np.any(idx1):
        first_obs_time = jds[idx1][0]
        phase_at_first = phases[idx1][0]
        queue.append({'name' : str(star['name']),
                      'time' : first_obs_time.datetime.time().strftime("%Hh %Mm %Ss"),
                      'phase' : phase_at_first,
                      'info' : "pre-mean",
                      'vmag' : star['magAvg']})

    if np.any(idx2):
        first_obs_time = jds[idx2][0]
        phase_at_first = phases[idx2][0]
        queue.append({'name' : str(star['name']),
                      'time' : first_obs_time.datetime.time().strftime("%Hh %Mm %Ss"),
                      'phase' : phase_at_first,
                      'info' : "post-mean",
                      'vmag' : star['magAvg']})

queue = Table(queue)
queue = queue['name','time','vmag','phase','info']
queue.sort('time')

fn = "queue_{0}.csv".format(day.datetime.date())
ascii.write(queue, os.path.join(output_path, fn), Writer=ascii.Basic, delimiter=',')