# coding: utf-8

""" Utility functions for handling astronomical time conversions """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.time import Time
from datetime import datetime, timedelta, time

from ..coordinates import sex_to_dec, dec_to_sex

__all__ = ["gmst_to_utc", "utc_to_gmst", "gmst_to_lmst", "lmst_to_gmst"]

def gmst_to_utc(t, utc_date):
    jd = int(Time(utc_date,scale='utc').jd) + 0.5
    
    S = jd - 2451545.0
    T = S / 36525.0
    T0 = 6.697374558 + (2400.051336*T) + (0.000025862*T**2)
    T0 = T0 % 24
    
    h = sex_to_dec((t.hour, t.minute, t.second))
    GST = (h - T0) % 24
    UT = GST * 0.9972695663
    
    tt = Time(jd, format='jd', scale='utc')
    dt = tt.datetime + timedelta(hours=UT)
    
    return Time(dt, scale='utc')

def utc_to_gmst(t):
    epoch = Time(datetime(2000,1,1,12,0,0), scale='utc')
    D = t - epoch
    D = (D.sec*u.second).to(u.day).value
    gmst = 18.697374558 + 24.06570982441908 * D
    return time(*dec_to_sex(gmst % 24, ms=True))

def gmst_to_lmst(t, longitude_w):
    gmst_hours = sex_to_dec((t.hour,t.minute,t.second,t.microsecond), ms=True)
    long_hours = longitude_w.to(u.hourangle).value
    lmst_hours = (gmst_hours + (24. - long_hours)) % 24.
    return time(*dec_to_sex(lmst_hours, ms=True))

def lmst_to_gmst(t, longitude_w):
    lmst_hours = sex_to_dec((t.hour,t.minute,t.second,t.microsecond), ms=True)
    long_hours = longitude_w.to(u.hourangle).value
    gmst_hours = (lmst_hours - (24. - long_hours)) % 24.
    return time(*dec_to_sex(gmst_hours, ms=True))