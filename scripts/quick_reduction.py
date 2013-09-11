# coding: utf-8

""" Script for quick-reducing spectra. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import glob

# Third-party
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

def coadd_files(filenames):
    """ Given a list of filenames containing FITS images, create a 
        master bias image.
    """
    
    master = None
    for filename in filenames:
        # Read in the first (only) HDU containing the image data
        hdu = fits.open(filename)[0]
        
        # Get image data from hdu
        img_data = hdu.data
        
        if master is None:
            master = img_data
        else:
            master += img_data
    
    master /= len(filenames)
    return master

def main():
    data_path = "/Users/adrian/Documents/GraduateSchool/Observing/2013-08_MDM/data/m090213"
    object_name = "SW And"
    
    flat_frames = []
    bias_frames = []
    object_files = []
    for filename in glob.glob(os.path.join(data_path, "*.fit*")):
        hdr = fits.getheader(filename, 0)
        object = hdr['OBJECT']
        
        if object.lower() == "flat":
            flat_frames.append(filename)
        elif object.lower() == "bias":
            bias_frames.append(filename)
        elif object.lower() == object_name.lower():
            object_files.append(filename)
        else:
            continue
    
    master_flat = coadd_files(flat_frames)
    master_bias = coadd_files(bias_frames)
    master_flat -= master_bias
    
    object_files = object_files[0:1]
    
    N_object = len(object_files)
    master_object = np.zeros((N_object,) + master_flat.shape)
    for ii,filename in enumerate(object_files):
        hdu = fits.open(filename)[0]
        img_data = hdu.data
        
        corrected = (img_data - master_bias) / master_flat
        master_object[ii] = corrected
    
    coadd = np.sum(master_object, axis=0) / N_object
    
    # filter out cosmic rays with a median filter
    #sigma = np.std(coadd)
    #CR_filter = coadd > 25.*sigma
    #coadd[CR_filter] = np.median(master_object, axis=0)[CR_filter]
    
    sum_down = np.sum(coadd[:,:400], axis=0)
    idx = np.argmax(sum_down)
    around_star = sum_down[idx-50:idx+50]
    
    # fit a Gaussian to summed 1D profile to find width
    def model(p, x):
        A, sigma, mu, b = p
        return A*np.exp(-0.5 * (x - mu)**2 / (2 * sigma**2)) + b
        
    def func(p, x, y):
        return y - model(p, x)
    
    x = np.arange(len(around_star))
    p_opt, ier = leastsq(func, x0=[1.,2.,idx, 1.], args=(x, around_star))
    A, sigma, mu, b = p_opt
    
    lidx = int(idx - 2*sigma)
    ridx = int(idx + 2*sigma)
    extracted = np.sum(coadd[:,lidx:ridx], axis=1)
    
    sky_l = np.sum(coadd[:,lidx-20:ridx-20], axis=1)
    sky_r = np.sum(coadd[:,lidx+20:ridx+20], axis=1)
    sky = (sky_l + sky_r)/2.
    
    plt.plot(extracted-sky)
    plt.show()
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", 
                    default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
                    default=False, help="Be quiet! (default = False)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    main()