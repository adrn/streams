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
import astropy.io.ascii as ascii

def read_LawMajewski_data():
    """ Read in particle position / velocity information for 1E5 particles from Law & Majewski's 2010 paper
        simulating the Sgr stream.
    """

    filename = "SgrTriax_DYN.dat"
    data = ascii.read(os.path.join("data", filename))
    return data

def main():
    linear_data = np.genfromtxt("data/LINEAR_RRab.txt", names=["ra", "dec", "objectID", "period", "rhjd0", "amp", "mag0", "template", "rExt", "magAvg"])
    law_data = read_LawMajewski_data()

    # Matplotlib accepts angles in *radians*, even though it displays in degrees...
    ra = [coord.Angle(ra, unit=u.degree) for ra in linear_data["ra"]]
    dec = [coord.Angle(dec, unit=u.degree) for dec in linear_data["dec"]]

    # Define mean metallicity for RR Lyrae, distance modulus
    Fe_H = -0.5
    distance_mod = linear_data["magAvg"] - (0.23*Fe_H + 0.93)

    sgr_coords = []
    for ii in range(len(linear_data)):
        distance_kpc = 10**(distance_mod[ii]/5. + 1.) / 1000.
        eq_coords = coord.ICRSCoordinates(ra[ii], dec[ii])
        sgr = eq_coords.transform_to(SgrCoordinates)
        sgr.distance = coord.Distance(distance_kpc, unit=u.kpc)
        sgr_coords.append(sgr)

    sgr_coords = np.array(sgr_coords)
    eq_coords = np.array([sgr.transform_to(coord.ICRSCoordinates) for sgr in sgr_coords])

    # TODO: once I get the coordinates right, figure out what cut to do from section 5.2 in Majewski et al. 2003
    #    (e.g. see page 1093, bottom right)
    #idx = np.array([sgr.z for sgr in sgr_coords]) > 11 # From Majewski
    idx = np.cos([sgr.Beta.radians for sgr in sgr_coords]) > 0.95
    sgr_plane_stars = sgr_coords[idx]
    eq_plane_stars = eq_coords[idx]

    # ---------------------------------
    # Make analog to Majewski's Fig. 12
    # ---------------------------------
    fig = plt.figure(figsize=(12,12))

    # His plots have strange wrappings, so setting the bounds here manually
    ra = [coord.Angle(eq.ra.degrees, unit=u.degree, bounds=(90, 450)).degrees for eq in eq_plane_stars]
    dec = [eq.dec.degrees for eq in eq_plane_stars]
    L = [coord.Angle(sgr.Lambda.degrees, unit=u.degree, bounds=(-180, 180)).degrees for sgr in sgr_plane_stars]
    B = [sgr.Beta.degrees for sgr in sgr_plane_stars]
    law_ra = [coord.Angle(rr, unit=u.degree, bounds=(90, 450)).degrees for rr in law_data["ra"]]
    law_dec = law_data["dec"]
    law_L = [coord.Angle(ll, unit=u.degree, bounds=(-180, 180)).degrees for ll in law_data["lambda"]]
    law_B = law_data["beta"]

    ax1 = fig.add_subplot(211)

    # Draw a line for the plane of Sgr
    line_L = np.linspace(-180, 180, 100)
    line_B = np.zeros(len(line_L))
    line_eq_coords = [SgrCoordinates(Ll,Bb,unit=(u.degree,u.degree)).icrs for Ll,Bb in zip(line_L,line_B)]
    #ax1.plot([coord.Angle(eq.ra.degrees, unit=u.degree, bounds=(90,450)).degrees for eq in line_eq_coords], [eq.dec.degrees for eq in line_eq_coords], color='r', marker='.', linestyle='none', alpha=0.5)

    ax1.scatter(ra, dec, color='b', s=2, alpha=0.3)
    ax1.scatter(law_ra, law_dec, color='r', s=1, alpha=0.1)
    ax1.set_xlim(450, 90)
    ax1.set_ylim(-60, 60)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\delta$")

    ax2 = fig.add_subplot(212)
    #ax2.plot(np.linspace(-180, 180, 100), np.zeros(100), color='r', alpha=0.5)
    ax2.scatter(L, B, color='b', s=2, alpha=0.3)
    ax2.scatter(law_L, law_B, color='r', s=1, alpha=0.1)
    ax2.set_xlim(180, -180)
    ax2.set_ylim(-60, 60)
    ax2.set_xlabel(r"$\Lambda_\odot$")
    ax2.set_ylabel(r"$B_\odot$")

    fig.subplots_adjust(hspace=0.1)

    fig.savefig("plots/sgr_with_particles_in_plane.png")

    # -------------------------------------------------------------
    # Make top-down plot in heliocentric Sgr cartesian coordinates
    # -------------------------------------------------------------
    rr_lyr_L = np.array([sgr.Lambda.radians for sgr in sgr_coords])
    rr_lyr_B = np.array([sgr.Beta.radians for sgr in sgr_coords])
    rr_lyr_r = np.array([sgr.distance.kpc for sgr in sgr_coords])

    X_sgr_sol = rr_lyr_r * np.cos(rr_lyr_L + radians(14.11)) * np.cos(rr_lyr_B)
    Y_sgr_sol = rr_lyr_r * np.sin(rr_lyr_L + radians(14.11)) * np.cos(rr_lyr_B)
    Z_sgr_sol = rr_lyr_r * np.sin(rr_lyr_B)

    # Majewski's cut
    #idx = (np.fabs(Z_sgr_sol) > 11) & (Z_sgr_sol > -30) & (Z_sgr_sol < 50)

    # Sgr X-Y plane
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.scatter(X_sgr_sol[idx], Y_sgr_sol[idx], color='b', alpha=0.5, edgecolor='none', s=2)

    law_idx = np.fabs(law_data["zsun"]) < 10
    ax.scatter(law_data["xsun"][law_idx], law_data["ysun"][law_idx], color='r', alpha=0.5, edgecolor='none', s=1)
    ax.plot(0., 0., color='#F4FF2B', marker='o', markersize=10)

    ax.set_xlim(-90, 65)
    ax.set_ylim(70, -80)
    ax.set_xlabel(r"$X_{Sgr,\odot}$")
    ax.set_ylabel(r"$Y_{Sgr,\odot}$")

    fig.savefig("plots/sgr_with_particles_top_down.png")

    return

    lambdas = np.array([sgr.Lambda.degrees for sgr in sgr_plane_stars])
    betas = np.array([sgr.Beta.degrees for sgr in sgr_plane_stars])

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