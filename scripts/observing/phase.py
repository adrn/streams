# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from argparse import ArgumentParser
from datetime import datetime
import os, sys

# Third-party
import numpy as np

def main(jd):

    grp_filename = "/Users/adrian/projects/streams/data/observing/rrl_targets_grp.txt"
    tbl = np.genfromtxt(grp_filename, names=True, dtype='|S10,f8,f8,f8,|S3,|S2,f8',
                        delimiter=',', filling_values=np.nan)

    not_done_idx = (tbl['done'] == 'n') & ~np.isnan(tbl['t0'])
    not_done_tbl = tbl[not_done_idx]

    for group in sorted(np.unique(not_done_tbl['group'])):
        group_tbl = not_done_tbl[not_done_tbl['group'] == group]

        for ii,row in enumerate(group_tbl):
            if ii == 0:
                print("-"*64)
                print("Group {}".format(group))

            phase = ((jd - row['t0']) % row['period']) / row['period'] * 12
            print("\t{} : {:.2f}".format(row['name'],phase))

if __name__ == "__main__":

    # Define parser object
    parser = ArgumentParser(description="Helper script for RR Lyrae observing")

    parser.add_argument("--jd", dest="jd", type=float, help="Julian date.", default=None)
    parser.add_argument("--group", dest="plot", action="store_true", default=False,
                        help="Plot or not")

    args = parser.parse_args()

    if args.jd is None:
        from astropy.time import Time
        now = datetime.utcnow()
        t = Time(now, scale='utc')
        jd = t.jd
        print("No JD specified -- using now: {}".format(jd))
    else:
        jd = args.jd
        print("Phases for JD: {}".format(jd))

    main(jd)