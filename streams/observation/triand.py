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
import numpy as np

# Project
from streams.coordinates import sex_to_dec
from streams.observation.time import gmst_to_utc, lmst_to_gmst
from streams.observation.rrlyrae import time_to_phase, phase_to_time
from streams.util import project_root

data_file = os.path.join(project_root, "data", "catalog", "TriAnd_RRLyr.txt")
stars = ascii.read(data_file,
                   converters={'objectID' : [ascii.convert_numpy(np.str)]},
                   header_start=0,
                   data_start=1,
                   delimiter=" ")

# Need to wrap so RA's go 22,23,24,0,1,etc.
ras = np.array(stars['ra'])
ras[ras > 90.] = ras[ras > 90.] - 360.
idx = np.argsort(ras)
stars = stars[idx]

names = ["TriAndRRL{0}".format(ii+1) for ii in range(len(stars))]
stars.add_column(Column(names, name='name'))

# Read in RR Lyrae standards
RRLyr_stds1 = ascii.read("/Users/adrian/Documents/GraduateSchool/Observing/Std RR Lyrae/nemec_RRLyrae.txt")
RRLyr_stds2 = ascii.read("/Users/adrian/Documents/GraduateSchool/Observing/Std RR Lyrae/bi-qing_for2011_RRLyr.txt", delimiter=',')
RRLyr_stds2.add_column(Column(RRLyr_stds2['hjd0_2450000'] + 2450000., name='rhjd0'))
RRLyr_stds2['ra'] = [coord.Angle(x, unit=u.hour).degree for x in RRLyr_stds2['ra_str']]
RRLyr_stds2['dec'] = [coord.Angle(x, unit=u.degree).degree for x in RRLyr_stds2['dec_str']]

standards = join(RRLyr_stds1, RRLyr_stds2, join_type='outer')
all_stars = join(stars, RRLyr_stds1, join_type='outer')
all_stars = join(all_stars, RRLyr_stds2, join_type='outer')
triand_stars = stars