# coding: utf-8

""" Classes for accessing various data related to this project. """

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

__all__ = ["read_linear", "read_quest", "read_catalina", "read_asas", "read_nsvs", "read_stripe82"]

def _make_npy_file(ascii_file, overwrite=False, ascii_kwargs=dict()):
    """ Make a .npy version of the given ascii file data.

        Parameters
        ----------
        ascii_file : str
            The full path to the ascii file to convert.
        overwrite : bool (optional)
            If True, will overwrite any existing npy files and regenerate
            using the latest ascii data.
        ascii_kwargs : dict
            A dictionary of keyword arguments to be passed to ascii.read().
    """

    filename, ext = os.path.splitext(ascii_file)
    npy_filename = filename + ".npy"

    if os.path.exists(npy_filename) and overwrite:
        os.remove(npy_filename)
    elif os.path.exists(npy_filename) and not overwrite:
        return npy_filename

    data = ascii.read(ascii_file, **ascii_kwargs)
    np.save(npy_filename, np.array(data))
    return npy_filename

def read_linear():
    """ Read in the LINEAR data -- RR Lyrae from the LINEAR survey, 
        sent to me from Branimir. 
    """
    txt_filename = os.path.join(project_root, "data", "catalog", \
                                "LINEAR_RRab.txt")
    data = ascii.read(txt_filename)
    
    # TODO: for V-band?!
    data.add_column(Column(np.zeros(len(data)), name="V"))
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
    #data.rename_column('RAJ2000', str("ra"))
    #data.rename_column('DEJ2000', str("dec"))
    data.add_column(Column(np.array(data["RAJ2000"]).astype(float), name=str("ra")))
    data.add_column(Column(np.array(data["DEJ2000"]).astype(float), name=str("dec")))
    
    #data.rename_column('Vmag', str("V"))
    data.add_column(Column(np.array(data["Vmag"]).astype(float), name=str("V")))
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
    data.rename_column('RA', str("ra"))
    data.rename_column('Dec', str("dec"))
    # has a V column
    
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
    data.rename_column('Vavg', str("V"))
    
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
    data.rename_column('RAJ2000', str("ra"))
    data.rename_column('DEJ2000', str("dec"))
    data.rename_column('Vmag', str("V"))
    
    return data

def read_stripe82():
    """ From: http://www.sdss.org/dr5/algorithms/sdssUBVRITransform.html
    
        V = g - 0.59*(g-r) - 0.01
    """
    txt_filename = os.path.join(project_root, "data", "catalog", \
                                "stripe82_rrlyr.txt")
    data = ascii.read(txt_filename, header_start=31, data_start=32)
    
    data.rename_column('RAdeg', str("ra"))
    data.rename_column('DEdeg', str("dec"))
    data.rename_column('Vmag', str("V"))
    
    return data