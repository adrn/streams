# coding: utf-8

""" General utilities for reducing data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import logging

# Third-party
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import custom_model_1d
from scipy.optimize import leastsq

# Create logger
logger = logging.getLogger(__name__)

#__all__ = ["parse_wavelength", "polynomial_fit", "gaussian_fit"]

@custom_model_1d
def GaussianPolynomial(x, b=0., m=0., log10_amp=4, stddev=1., mean=0):
    return models.Linear1DModel.eval(x, slope=m, intercept=b) + \
           models.Gaussian1DModel.eval(x, amplitude=10**log10_amp, \
                                       stddev=stddev, mean=mean)

def gaussian_fit(x, y, order=0, **p0):
    """ Fit a Gaussian + polynomial to the data.

        Parameters
        ----------
        x : array_like
        y : array_like
        order : int (optional)
            The order of the Polynomial model. Defaults to constant (order=0).
        **p0
            Initial conditions for the model fit.

    """
    if len(x) != len(y):
        raise ValueError("x and y must have the sample shape!")

    # TODO: when astropy.modeling allows combining models, fix this
    if order > 0:
        raise NotImplementedError()

    #g = custom_model_1d(_gaussian_constant_model)
    g = GaussianPolynomial()
    default_p0 = dict(b=min(y),
                      m=0.,
                      log10_amp=np.log10(max(y)),
                      stddev=0.5,
                      mean=float(np.mean(x)))

    for k,v in default_p0.items():
        setattr(g, k, p0.get(k,v))

    fit = fitting.NonLinearLSQFitter(g)
    fit(x, y)

    return g

def polynomial_fit(x, y, order=3):
    """ Fit a polynomial of the specified order to the given
        x and y data, return an astropy.modeling.Model fit to
        the data.
    """

    if len(x) != len(y):
        raise ValueError("x and y must have the sample shape!")

    p = models.Polynomial1DModel(order)
    fit = fitting.LinearLSQFitter(p)
    fit(x, y)

    return p

def parse_wavelength(wvln, default_unit=u.angstrom):
    """ Parse a wavelength string from raw_input and return
        an astropy.units.Quantity object.
    """

    try:
        wvln_value, wvln_unit = wvln.split()
    except ValueError:
        # assume angstroms
        wvln_value = wvln
        wvln_unit = default_unit

    return float(wvln_value) * u.Unit(wvln_unit)

def find_all_imagetyp(path, imagetyp):
    """ Find all FITS files in the given path with the IMAGETYP
        header keyword equal to whatever is specified.

        Parameters
        ----------
        path : str
            Path to a bunch of FITS files.
        imagetype : str
            The desired IMAGETYP.
    """

    files = []
    for filename in glob.glob(os.path.join(path, "*.fit*")):
        hdr = fits.getheader(filename,0)
        if hdr["IMAGETYP"] == imagetyp:
            files.append(filename)

    return files