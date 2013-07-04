# coding: utf-8

""" Compile data from:
    1) Photometric catalogs of RR Lyrae from Stripe 82 and QUEST
        QUEST:
        Stripe 82: 
        

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt

# Project
from streams.util import project_root
from streams.observation.rrlyrae import *
from streams.data import add_sgr_coordinates, read_quest
from streams.data.simulation import lm10_particle_data

if __name__ == "__main__":
    # TODO: need sample for Allyson's observing run in 2 weeks!!
    quest_data = read_quest()
    quest_data = add_sgr_coordinates(quest_data)
    
    # Distance < 70 kpc, and only particles within 10 kpc of the Sgr plane
    idx = (np.fabs(quest_data['sgr_plane_dist']) < 15.)
    quest_data = quest_data[idx]
    
    # Read in LM10 simulation data
    lm10 = lm10_particle_data(expr="(Pcol>-1) & (Pcol<6) & (abs(Lmflag)==1)")
    south_wrap = lm10_particle_data(expr="(Pcol>-1) & (Lmflag==1) & (Pcol<7) & "
                                         "(Lambda < 240) & (Lambda > 60)")
    north_wrap = lm10_particle_data(expr="(Pcol>-1) & (Lmflag==-1) & (Pcol<7) & "
                                         "(Lambda > 200) & (Lambda < 340)")
    
    # Make a plot for K. Vivas / KVJ
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='polar')
    
    ax.scatter(np.radians(lm10['Lambda']), lm10['dist'],
               edgecolor='none', color='#666666', marker='.', alpha=0.1)
    
    ax.scatter(np.radians(north_wrap['Lambda']), north_wrap['dist'],
               edgecolor='none', color='#5E3C99', marker='.', alpha=0.25)
    ax.scatter(np.radians(south_wrap['Lambda']), south_wrap['dist'],
               edgecolor='none', color='#5E3C99', marker='.', alpha=0.25)
    
    # Plot data without spectra
    ax.scatter(np.radians(quest_data['Lambda'][quest_data['Vgsr'].mask]), 
               quest_data['dist'][quest_data['Vgsr'].mask],
               edgecolor='none', color='#E66101', marker='o', s=20, alpha=0.5)
    
    # Plot data with spectra
    ax.scatter(np.radians(quest_data['Lambda'][quest_data['has_spectrum']]), 
               quest_data['dist'][quest_data['has_spectrum']],
               edgecolor='none', color='#1B7837', marker='^', s=50, alpha=0.7)
    
    ax.set_theta_direction(-1)
    ax.set_ylim(0,70)
    ax.set_xlabel(r"$\Lambda_{\odot}$", fontsize=24)
    ax.set_ylabel(r"[kpc]", fontsize=12, rotation='horizontal')
    
    plt.show()