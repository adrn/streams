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
from streamteam.observation import sex_to_dec, gmst_to_utc, lmst_to_gmst
from streamteam.observation.rrlyrae import time_to_phase, phase_to_time

#########################################################################
# Set these

# - the target list must contain, at least, columns 'name' for the star name,
#       'ra' and 'dec' for sky coordinates, 'period' for the period in days,
#       'hjd0' for the HJD (MJD) of the peak of the pulsation, and
#       'Vmag' for the V-band magnitude
target_list_filename = "/Users/adrian/projects/streams/data/observing/triand.txt"

# path to write files and plots to
output_path = ""

#
#########################################################################

kitt_peak_longitude = (111. + 35/60. + 40.9/3600)*u.deg
def tcs_list(stars):
    """ Given a table of stars, ouput a list to be fed in to the TCS """

    names = []
    ras = []
    decs = []
    epoch = []
    for ii,star in enumerate(stars):
        names.append(star['name'])
        ra = coord.Longitude(star['ra']*u.deg)
        dec = coord.Latitude(star['dec']*u.deg)

        ras.append(ra.to_string(unit=u.hour, sep=' ', pad=True, precision=1))
        decs.append(dec.to_string(unit=u.deg, sep=' ', alwayssign=True, precision=1))
        epoch.append(2000.0)

    t = Table()
    t.add_column(Column(names, name='name'))
    t.add_column(Column(ras, name='ra'))
    t.add_column(Column(decs, name='dec'))
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

    hour_angles = [-buffer_time.to(u.hour).value,buffer_time.to(u.hour).value]*u.hourangle

    jds = []
    for ha in hour_angles:
        lst = day.datetime + timedelta(hours=(ha + ra).to(u.hourangle).value)
        gmst = lmst_to_gmst(lst.time(), kitt_peak_longitude)
        utc = gmst_to_utc(gmst, utc_day)
        jds.append(utc.jd)

    return Time(jds, scale='utc', format='jd')

# def open_finder_charts(stars):
#     for ii,star in enumerate(stars):
#         url = "http://ptf.caltech.edu/cgi-bin/ptf/variable/fc.cgi?name={1}{0}&ra={ra}&dec={dec}&bigwidth=300.000000&bigspan=19.000000&zoomwidth=150.000000&zoomspan=8.000000"
#         print(url.format(ii+1, star_name_prefix, ra=star['ra'], dec=star['dec']))
#         os.system("open '{0}'".format(url.format(ii+1, ra=star['ra'], dec=star['dec'])))

# read the target list
stars = ascii.read(target_list_filename)

# output list of all targets to be added to the TCS
tcs = tcs_list(stars)
fn = os.path.join(output_path, "tcs_list.txt")
ascii.write(tcs, fn, Writer=ascii.Basic)
with open(fn, 'r') as f:
    d = "".join(f.readlines()[1:]).replace('"', '')

with open(fn, 'w') as f:
    f.write(d)

# Create a queue for the given day
queue = []

_date = map(int, sys.argv[1].split("-"))
day = Time(datetime(*_date), scale='utc')

phase_plot_path = os.path.join(output_path, str(day.datetime.date()))
if not os.path.exists(phase_plot_path):
    os.mkdir(phase_plot_path)

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

    if jd1 > jd2:
        jd2 += 1.

    obs_window = Time([jd1, jd2], format='jd', scale='utc')
    period = star['period']*u.day
    t0 = Time(star['hjd0'], scale='utc', format='mjd')

    # Now we want to see if at what time the phase of the pulsation is around
    #   0.4 and 0.7
    #optimal_phases = np.array([0.4, 0.7])
    #optimal_phase_times = phase_to_time(optimal_phases, day=day, t0=t0, period=period)
    #print(optimal_phase_times.datetime, obs_window.datetime)
    #continue

    # Here instead we'll look at the possible times we can observe
    step = 1.*u.minute
    jds = Time(np.arange(jd1, jd2, step.to(u.day).value), format='jd', scale='utc')

    ut_hours = np.array([sex_to_dec((jd.datetime.hour,jd.datetime.minute,jd.datetime.second)) for jd in jds])
    phases = time_to_phase(jds, period=period, t0=t0)

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(ut_hours, phases, marker='o', linestyle='none')
    ax.set_xlim(1.5, 13.)
    ax.set_ylim(0., 1.)
    ax.set_xlabel("UT time")
    ax.text(5., 0.9, star['name'])
    fig.savefig(os.path.join(phase_plot_path, "{0}.png".format(star['name'])))

    idx1 = (phases > 0.1) & (phases < 0.5) & \
           (ut_hours < 13.) & (ut_hours > 2.)

    idx2 = (phases >= 0.5) & (phases < 0.8) & \
           (ut_hours < 13.) & (ut_hours > 2.)

    if not np.any(idx1) and not np.any(idx2):
        print("Skipping {0}, no suitable observing time."\
                .format(star['name']))
        continue

    if np.any(idx1):
        first_obs_time = jds[idx1][0]
        phase_at_first = phases[idx1][0]
        queue.append({'name' : str(star['name']),
                      'time' : first_obs_time.datetime.time().strftime("'%H:%M:%S'"),
                      'phase' : phase_at_first,
                      'info' : "pre-mean",
                      'vmag' : star['Vmag']})

    if np.any(idx2):
        first_obs_time = jds[idx2][0]
        phase_at_first = phases[idx2][0]
        queue.append({'name' : str(star['name']),
                      'time' : first_obs_time.datetime.time().strftime("'%H:%M:%S'"),
                      'phase' : phase_at_first,
                      'info' : "post-mean",
                      'vmag' : star['Vmag']})

queue = Table(queue)
queue = queue['name','time','vmag','phase','info']
queue.sort('time')

fn = "queue_{0}.csv".format(day.datetime.date())
ascii.write(queue, os.path.join(output_path, fn), Writer=ascii.Basic, delimiter=',')