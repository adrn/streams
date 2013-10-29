# coding: utf-8

""" General utilities for reducing data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
#import logging

# Third-party
import numpy as np
import astropy.units as u
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import Custom1DModel
from scipy.optimize import leastsq

# Create logger
#logger = logging.getLogger(__name__)

__all__ = ["spectral_line_model", "spectral_line_erf", "parse_wavelength",
           "polynomial_fit", "line_list", "gaussian_fit"]

def _gaussian_constant_model(x, c=0., log10_amp=4, stddev=1., mean=0):
    return models.Const1DModel.eval(x, amplitude=c) + \
           models.Gaussian1DModel.eval(amplitude=10**log10_amp, \
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

    g = Custom1DModel(_gaussian_constant_model)
    default_p0 = dict(c=min(y),
                      log10_amp=np.log10(max(y)),
                      stddev=0.5,
                      mean=float(np.mean(x))]
    for k,v in default_p0.items():
        setattr(g,k,p0.get(k,v))

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

def line_list(name):
    """ Read in a list of wavelengths for a given arc lamp.

        TODO: this could be so much better. Need some kind of registry?
        TODO: is there a full list for Hg Ne somewhere?
    """

    from . import obs_path

    if name.replace(" ", "").lower() == "hgne":
        fn = os.path.join(obs_path, "MDM 2.4m", "line lists", "Hg_Ne.txt")
        lines = np.loadtxt(fn)
    else:
        raise ValueError("No list for {0}".format(name))

    return lines