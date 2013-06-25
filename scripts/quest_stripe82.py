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
from astropy.table import Table, Column, vstack, join
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt

# Project
from streams.util import project_root
from streams.observation.rrlyrae import *
from streams.data import add_sgr_coordinates
from streams.data.simulation import lm10_particle_data

def read_quest():
    """ Read in the QUEST data -- RR Lyrae from the QUEST survey,
        Vivas et al. 2004. 
        
        - Photometry from:
            http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/127/1158
        - Spectral data from:
            http://iopscience.iop.org/1538-3881/129/1/189/fulltext/204289.tables.html
            Spectroscopy of bright QUEST RR Lyrae stars (Vivas+, 2008)
            
    """
    phot_filename = os.path.join(project_root, "data", "catalog", \
                                 "quest_vivas2004_phot.tsv")
    phot_data = ascii.read(phot_filename, delimiter="\t", data_start=3)
    
    # With more spectral data, add here
    vivas2004_spec = ascii.read(os.path.join(project_root, "data", 
                                             "catalog", "quest_vivas2004_spec.tsv"),
                                delimiter="\t")
                                
    vivas2008_spec = ascii.read(os.path.join(project_root, "data", 
                                             "catalog", "quest_vivas2008_spec.tsv"),
                                delimiter="\t", data_start=3)
    
    spec_data = vstack((vivas2004_spec, vivas2008_spec))
    all_data = join(left=phot_data, right=spec_data, keys=['[VZA2004]'], join_type='outer')
    
    new_columns = dict()
    new_columns['ra'] = []
    new_columns['dec'] = []
    new_columns['V'] = []
    new_columns['dist'] = []
    for row in all_data:
        icrs = coord.ICRSCoordinates(row["_RAJ2000_1"], 
                                     row["_DEJ2000_1"], unit=(u.degree,u.degree))
        new_columns['ra'].append(icrs.ra.degrees)
        new_columns['dec'].append(icrs.dec.degrees)
        
        v1 = row['Vmag_1']
        v2 = row['Vmag_2']
        if v1 != None:
            new_columns['V'].append(v1)
        else:
            new_columns['V'].append(v2)
        
        if row['Dist'] != None:
            d = row['Dist']
        else:
            d = rrl_photometric_distance(new_columns['V'][-1], -1.5)
        new_columns['dist'].append(d)
    
    for name,data in new_columns.items():
        all_data.add_column(Column(data, name=name))
    
    all_data["ra"].units = u.degree
    all_data["dec"].units = u.degree
    all_data["dist"].units = u.kpc
    
    all_data.remove_column('Lambda')
    all_data.remove_column('Beta')
    
    has_spectrum = np.logical_not(np.array(all_data['Vgsr'].mask))
    all_data.add_column(Column(has_spectrum, name='has_spectrum'))
    
    return all_data

if __name__ == "__main__":
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