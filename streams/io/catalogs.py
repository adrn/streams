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
from astropy.table import Table, Column, vstack, join

# Project
from ..util import project_root
from ..observation.rrlyrae import *

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
    
    data["ra"].units = u.degree
    data["dec"].units = u.degree
    data["dist"].units = u.kpc
    
    return data

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
    
    vivas2008_spec.rename_column('HJD0', 'HJD')
    spec_data = vstack((vivas2004_spec, vivas2008_spec))
    all_data = join(left=phot_data, right=spec_data, keys=['[VZA2004]'], join_type='outer')
    
    new_columns = dict()
    new_columns['ra'] = []
    new_columns['dec'] = []
    new_columns['V'] = []
    new_columns['dist'] = []
    new_columns['Type'] = []
    new_columns['Per'] = []
    new_columns['HJD'] = []
    for row in all_data:
        if not isinstance(row["_RAJ2000_1"], np.ma.core.MaskedConstant):
            icrs = coord.ICRSCoordinates(row["_RAJ2000_1"], 
                                         row["_DEJ2000_1"], 
                                         unit=(u.degree,u.degree))
        elif not isinstance(row["_RAJ2000_2"], np.ma.core.MaskedConstant):
            icrs = coord.ICRSCoordinates(row["_RAJ2000_2"], 
                                         row["_DEJ2000_2"], 
                                         unit=(u.degree,u.degree))
        else:
            raise TypeError()
        
        new_columns['ra'].append(icrs.ra.degrees)
        new_columns['dec'].append(icrs.dec.degrees)
        
        if not isinstance(row["Type_1"], np.ma.core.MaskedConstant):
            new_columns['Type'].append(row['Type_1'])
        elif not isinstance(row["Type_2"], np.ma.core.MaskedConstant):
            new_columns['Type'].append(row['Type_2'])
        else:
            raise TypeError()
        
        if not isinstance(row["Per_1"], np.ma.core.MaskedConstant):
            new_columns['Per'].append(row['Per_1'])
        elif not isinstance(row["Per_2"], np.ma.core.MaskedConstant):
            new_columns['Per'].append(row['Per_2'])
        else:
            raise TypeError()
        
        if not isinstance(row["HJD_1"], np.ma.core.MaskedConstant):
            new_columns['HJD'].append(row['HJD_1'])
        elif not isinstance(row["HJD_2"], np.ma.core.MaskedConstant):
            new_columns['HJD'].append(row['HJD_2'])
        else:
            raise TypeError()
        
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
    data.add_column(Column(rrl_photometric_distance(np.array(data['V']), -1.5).to(u.kpc).value, 
                           name="dist", unit=u.kpc))
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
    
    data["ra"].units = u.degree
    data["dec"].units = u.degree
    data["dist"].units = u.kpc
    
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
    
    data["ra"].units = u.degree
    data["dec"].units = u.degree
    data["dist"].units = u.kpc
    
    return data

def read_stripe82():
    """ From: http://www.sdss.org/dr5/algorithms/sdssUBVRITransform.html
    
        V = g - 0.59*(g-r) - 0.01
    """
    txt_filename = os.path.join(project_root, "data", "catalog", \
                                "stripe82_rrlyr.tsv")
    data = ascii.read(txt_filename, delimiter='\t')
    
    data.add_column(Column(np.array(data["RAJ2000"]).astype(float), 
                           name=str("ra")))
    data.add_column(Column(np.array(data["DEJ2000"]).astype(float), 
                           name=str("dec")))
    data.add_column(Column(np.array(data["Vmag"]).astype(float), 
                           name=str("V")))
    data.add_column(Column(np.array(data["Dist"]).astype(float), 
                           name="dist", units=u.kpc))
    
    data["ra"].units = u.degree
    data["dec"].units = u.degree
    data["dist"].units = u.kpc
    
    return data