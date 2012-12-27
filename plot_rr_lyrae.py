# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import math

# Third-party
import numpy as np
from numpy import sin, cos, radians, degrees
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from streams.coordinates import SgrCoordinates
import astropy.coordinates as coord
import astropy.units as u

def main():
    linear_data = np.genfromtxt("data/LINEAR_RRab.txt", names=["ra", "dec", "objectID", "period", "rhjd0", "amp", "mag0", "template", "rExt", "magAvg"])

    # Matplotlib accepts angles in *radians*, even though it displays in degrees...
    ra = [coord.Angle(ra, unit=u.degree) for ra in linear_data["ra"]]
    dec = [coord.Angle(dec, unit=u.degree) for dec in linear_data["dec"]]

    # Define mean metallicity for RR Lyrae, distance modulus
    Fe_H = -1.5
    distance_mod = linear_data["magAvg"] - (0.23*Fe_H + 0.93)

    sgr_coords = []
    for ii in range(len(linear_data)):
        distance_kpc = 10**(distance_mod[ii]/5. + 1.) / 1000.
        eq_coords = coord.ICRSCoordinates(ra[ii], dec[ii])
        sgr = eq_coords.transform_to(SgrCoordinates)
        sgr.distance = coord.Distance(distance_kpc, unit=u.kpc)
        sgr_coords.append(sgr)

    sgr_coords = np.array(sgr_coords)

    # TODO: once I get the coordinates right, figure out what cut to do from section 5.2 in Majewski et al. 2003
    #    (e.g. see page 1093, bottom right)
    #idx = np.cos([sgr.Beta.radians for sgr in sgr_coords]) > 0.95
    idx = np.array([sgr.z for sgr in sgr_coords]) > 11 # From Majewski
    sgr_plane_stars = sgr_coords[idx]

    lambdas = np.array([sgr.Lambda.degrees for sgr in sgr_plane_stars])
    betas = np.array([sgr.Beta.degrees for sgr in sgr_plane_stars])

    # Sgr X-Y plane
    plt.figure(figsize=(12,12))
    plt.scatter([sgr.x-8. for sgr in sgr_plane_stars], [sgr.y for sgr in sgr_plane_stars], color='k', marker='.', alpha=0.5, edgecolor='none', s=20.)
    plt.plot(-8., 0., color='#F4FF2B', marker='o', markersize=15)
    plt.xlim(-90, 65)
    plt.ylim(70, -80)
    plt.xlabel(r"$X_{Sgr,GC}$")
    plt.ylabel(r"$Y_{Sgr,GC}$")
    plt.show()

    return

    # Bin in 2d by position, compute median distance, plot 2d histogram
    bin_size = 3.
    lambda_bins = np.arange(lambdas.min(), lambdas.max(), bin_size)
    beta_bins = np.arange(betas.min(), betas.max(), bin_size)

    distance_map = np.zeros((len(lambda_bins), len(beta_bins)))
    for ii,lambda_bin_left in enumerate(lambda_bins):
        lambda_bin_right = lambda_bin_left + bin_size
        for jj,beta_bin_left in enumerate(beta_bins):
            beta_bin_right = beta_bin_left + bin_size

            idx = (lambdas > lambda_bin_left) & (lambdas < lambda_bin_right) &\
                  (betas > beta_bin_left) & (betas < beta_bin_right)

            try:
                #dist = np.median([sgr.distance.kpc for sgr in sgr_plane_stars[idx]])
                dist = np.mean([sgr.distance.kpc for sgr in sgr_plane_stars[idx]])
            except ValueError:
                continue

            if np.isnan(dist):
                distance_map[ii,jj] = 0.
            else:
                distance_map[ii,jj] = dist

    # plot Sgr longitude, latitude with color representing distance
    X,Y = np.meshgrid(lambda_bins, beta_bins)
    Z = distance_map.T
    plt.figure(figsize=(15,5))
    img = plt.pcolor(X, Y, Z, cmap=cm.Blues)
    plt.colorbar(img)
    plt.xlim(plt.xlim()[1], plt.xlim()[0])
    plt.xlabel(r"$\Lambda$")
    plt.ylabel(r"$\beta$")
    plt.show()

    return

    #H, xedges, yedges = np.histogram2d(lambdas, betas, bins=(lambda_bins, beta_bins))
    #plt.imshow(H.T, interpolation="nearest")
    #plt.show()
    #return

    # Custom hammer projection
    lambdas -= 180.
    x = 2*np.sqrt(2)*np.cos(np.radians(betas))*np.sin(np.radians(lambdas/2.)) / np.sqrt(1 + np.cos(np.radians(betas))*np.cos(np.radians(lambdas/2.)))
    y = np.sqrt(2)*np.sin(np.radians(betas)) / np.sqrt(1 + np.cos(np.radians(betas))*np.cos(np.radians(lambdas/2.)))
    dists = np.array([sgr.distance.kpc for sgr in sgr_plane_stars])

    # Sgr Longitude vs. Distance
    #plt.plot([sgr.Lambda.degrees for sgr in sgr_plane_stars], dists, 'k.')
    #plt.show()

    # Bin by distance
    bin_size = 5.
    for bin_left in np.arange(0., 50., bin_size):
        bin_right = bin_left+bin_size

        idx = (dists > bin_left) & (dists < bin_right)
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111)
        ax.scatter(np.degrees(x[idx]),
                   np.degrees(y[idx]),
                   s=15, edgecolor='none')
        #                   c=dists[idx],

        ax.set_title("Distance: {0}-{1} kpc".format(bin_left, bin_right))
        ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])
        fig.savefig("plots/rrlyrae/linear_dist_{0}_{1}.png".format(bin_left, bin_right))

if __name__ == "__main__":
    main()
    #test_transform()