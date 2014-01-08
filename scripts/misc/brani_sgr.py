# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
from astropy.coordinates import Angle
from astropy.io import ascii
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def lambda_vs_dist(lm10, Nbins=30):
    """ Define interpolating functions for the leading and trailing
        wrap of Sgr debris to compute distance as a function of the
        Sgr longitude: Lambda.

        Parameters
        ----------
        lm10 :
    """

    # for each arm (trailing, leading), create a function that accepts a
    #   Sgr longitude (Lambda) and returns the heliocentric distance to the
    #   median position of the stream at that longitude
    interp_funcs = dict()
    for lmflag in [1,-1]:
        # select only particles from either leading (lmflag=1) or
        #   trailing (lmflag=-1) arm
        wrap = lm10[lm10['Lmflag'] == lmflag]
        bins = np.linspace(wrap["Lambda"].min(), wrap["Lambda"].max(), Nbins)

        median_dist = []
        dist_scatter = []
        for ii in range(Nbins-1):
            binL = bins[ii]
            binR = bins[ii+1]

            idx = (wrap["Lambda"] > binL) & (wrap["Lambda"] < binR)
            m = np.median(wrap[idx]["dist"])
            # s = np.std(wrap[idx]["dist"]) # can also store the LOS spread..
            median_dist.append(m)

        bin_centers = (bins[1:]+bins[:-1])/2.
        dist_func = interp1d(bin_centers, median_dist,
                             kind='cubic', bounds_error=False)
        interp_funcs[lmflag] = dist_func

    # you can comment this out if you don't want to deal with astropy units
    #   and just return interp_funcs[1], interp_funcs[-1]
    def leading(L):
        try:
            d = interp_funcs[1](L.degree)
        except AttributeError:
            raise ValueError("Input longitude must have angle-like units!")
        return d*u.kpc

    def trailing(L):
        try:
            d = interp_funcs[-1](L.degree)
        except AttributeError:
            raise ValueError("Input longitude must have angle-like units!")
        return d*u.kpc

    return leading, trailing

def main():
    # Read in the LM10 simulation data -- you can download this file here:
    #   http://www.astro.virginia.edu/~srm4n/Sgr/SgrTriax_DYN.dat.gz
    lm10_data_file = "/path/to/SgrTriax_DYN.dat"
    #lm10_data_file = ("/Users/adrian/projects/streams"
    #                  "/data/simulation/LM10/SgrTriax_DYN.dat")
    lm10_data = ascii.read(lm10_data_file)

    # select out only the first leading / trailing wraps, unbound particles
    lm10_data = lm10_data[np.fabs(lm10_data["Lmflag"]) == 1]
    lm10_data = lm10_data[(lm10_data["Pcol"]<8) & (lm10_data["Pcol"]>-1)]

    # need to rename some columns
    for x in "xyz":
        lm10_data.rename_column("{}gc".format(x), x)

    for v,w in zip("uvw", ("vx","vy","vz")):
        lm10_data.rename_column("{}".format(v), w)

    leading,trailing = lambda_vs_dist(lm10_data, Nbins=30)

    # now we can evaluate the median distance to the stream at any Longitude:
    L = Angle(np.linspace(0,360)*u.degree)

    fig,ax = plt.subplots(1,1,figsize=(8,8),
                          subplot_kw=dict(projection="polar"))

    ax.plot(np.radians(lm10_data["Lambda"]), lm10_data["dist"], marker='.',
            linestyle='none', alpha=0.1, color='#555555')
    ax.plot(L.radian, leading(L), marker=None, linestyle='-',
            linewidth=2., alpha=0.75, color='#0571B0')
    ax.plot(L.radian, trailing(L), marker=None, linestyle='-',
            linewidth=2., alpha=0.75, color='#CA0020')

    ax.set_theta_direction(-1)
    ax.set_ylim(0,70)
    plt.show()

if __name__ == "__main__":
    main()