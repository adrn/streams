# coding: utf-8

""" Convenience functions for reading in catalogs of RR Lyrae stars """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy.table import Table, Column
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import astropy.coordinates as coord
import astropy.units as u

# Project
from ..util import project_root

__all__ = ["read_linear", "read_quest", "read_catalina", "read_asas", \
           "read_nsvs", "read_stripe82"]
           
def read_linear():
    """ Read in the LINEAR data -- RR Lyrae from the LINEAR survey, 
        sent to me from Branimir. 
    """
    txt_filename = os.path.join(project_root, "data", "catalog", \
                                "LINEAR_RRab.txt")
    data = ascii.read(txt_filename)
    
    # Assuming a mean halo metallicity of -1.5 dex -- from Chaboyer 1999
    M = 0.23*(-1.5) + 0.93
    mu = data['magAvg'] - M
    dist = (10**(mu/5. + 1)*u.pc).to(u.kpc)
    data.add_column(Column(dist, name="dist", units=u.kpc))
    
    return data

def read_quest():
    """ Read in the QUEST data -- RR Lyrae from the QUEST survey,
        Vivas et al. 2004. 
        
        - Covers Stripe 82
    """
    fits_filename = os.path.join(project_root, "data", "catalog", \
                                "quest_vivas2004_RRL.fits")
    hdulist = fits.open(fits_filename)
    tb_data = np.array(hdulist[1].data)
    data = Table(tb_data)
    
    # Map RA/Dec columns to easier names
    data.add_column(Column(np.array(data["RAJ2000"]).astype(float), 
                           name=str("ra")))
    data.add_column(Column(np.array(data["DEJ2000"]).astype(float), 
                           name=str("dec")))
    data.add_column(Column(np.array(data["Vmag"]).astype(float), 
                           name=str("V")))
    data.add_column(Column(rrl_photometric_distance(data['V'], -1.5), 
                           name="dist", units=u.kpc))
    
    return data

def read_catalina():
    """ Read in the Catalina data -- RR Lyrae from the Catalina All Sky Survey,
        
        http://iopscience.iop.org/0004-637X/763/1/32/pdf/apj_763_1_32.pdf
        http://nesssi.cacr.caltech.edu/DataRelease/RRL.html
        
    """
    txt_filename = os.path.join(project_root, "data", "catalog", \
                                "Catalina_RRLyr.txt")
    data = ascii.read(txt_filename)    
    
    # Map RA/Dec columns to easier names
    data.add_column(Column(np.array(data["RA"]).astype(float), 
                           name=str("ra")))
    data.add_column(Column(np.array(data["Dec"]).astype(float), 
                           name=str("dec")))
    # has a V column
    data.add_column(Column(rrl_photometric_distance(data['V'], -1.5), 
                           name="dist", units=u.kpc))
    return data

def read_asas():
    """ Read in the ASAS data -- RR Lyrae from the ASAS
        http://adsabs.harvard.edu/abs/2009AcA....59..137S
        
    """
    tsv_filename = os.path.join(project_root, "data", "catalog", \
                                "ASAS_RRab.tsv")
    data = ascii.read(tsv_filename, data_start=59)
    
    ras,decs = [],[]
    for radec in data["ASAS"]:
        try:
            neg = 1.
            ra,dec = radec.split("+")
        except:
            neg = -1
            ra,dec = radec.split("-")
        
        ra = float(ra[:2]) + float(ra[2:4])/60. + float(ra[4:])/3600.
        dec = neg*(float(dec[:2]) + float(dec[2:])/60.)
        
        fk5 = coord.FK5Coordinates(ra, dec, unit=(u.hour,u.degree))
        ras.append(fk5.ra.degrees)
        decs.append(fk5.dec.degrees)
    
    # Map RA/Dec columns to easier names
    data.add_column(Column(ras, name="ra", dtype=float))
    data.add_column(Column(decs, name="dec", dtype=float))
    data.add_column(Column(np.array(data['Vavg']).astype(float), name=str("V")))
    data.add_column(Column(rrl_photometric_distance(data['V'], -1.5), 
                           name="dist", units=u.kpc))    
    return data

def read_nsvs():
    """ Read in the NSVS data -- RR Lyrae from the NSVS
        http://adsabs.harvard.edu/abs/2004AJ....127.2436W
        http://vizier.cfa.harvard.edu/viz-bin/VizieR-4
        
    """
    
    tsv_filename = os.path.join(project_root, "data", "catalog", \
                                "NSVS_RRab.tsv")
    data = ascii.read(tsv_filename, data_start=52)
    
    # Map RA/Dec columns to easier names
    data.add_column(Column(np.array(data["RAJ2000"]).astype(float), 
                           name=str("ra")))
    data.add_column(Column(np.array(data["DEJ2000"]).astype(float), 
                           name=str("dec")))
    data.add_column(Column(np.array(data["Vmag"]).astype(float), 
                           name=str("V")))
    data.add_column(Column(rrl_photometric_distance(data['V'], -1.5), 
                           name="dist", units=u.kpc))

    return data

def read_stripe82():
    """ From: http://www.sdss.org/dr5/algorithms/sdssUBVRITransform.html
    
        V = g - 0.59*(g-r) - 0.01
    """
    txt_filename = os.path.join(project_root, "data", "catalog", \
                                "stripe82_rrlyr.txt")
    data = ascii.read(txt_filename, header_start=31, data_start=32)
    
    data.add_column(Column(np.array(data["RAdeg"]).astype(float), 
                           name=str("ra")))
    data.add_column(Column(np.array(data["DEdeg"]).astype(float), 
                           name=str("dec")))
    data.add_column(Column(np.array(data["Vmag"]).astype(float), 
                           name=str("V")))
    data.add_column(Column(rrl_photometric_distance(data['V'], -1.5), 
                           name="dist", units=u.kpc))
    
    return data