# coding: utf-8

""" Classes for dealing with CCD data. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import astropy.units as u
import numpy as np

# Create logger
logger = logging.getLogger(__name__)

class CCD(object):

    def __init__(self, shape, gain, read_noise, dispersion_axis=0):
        """ Represents a CCD detector. You should set the following
            additional attributes:

                data_mask : a boolean mask with shape
                    'data[readout_mask].shape' that picks out the region
                    to be used for science;
                overscan_mask : a boolean mask with shape
                    'data[readout_mask].shape' that designates the overscan.

            TODO: gain and read_noise should have units?

            Parameters
            ----------
            shape : tuple
                Number of pixels along each axis.
            gain : numeric
            read_noise : numeric
            dispersion_axis : int (optional)
                Defaults to axis=0. The dispersive axis, e.g., wavelength.
        """

        self.shape = tuple(shape)
        if len(self.shape) != 2:
            raise ValueError("'shape' must be a 2 element iterable.")

        self.gain = float(gain)
        self.read_noise = float(read_noise)
        self.dispersion_axis = int(dispersion_axis)

        # can define named sub-regions of the detector
        self.regions = dict()

    def __getitem__(self, *slices):
        return CCDRegion(self, *slices)

    def overscan_subtract(self, data):
        """ Subtract the overscan region from this data. Note that the
            shape of the returned data will be different from input.
        """
        overscan = data[self.regions["overscan"]]
        overscan_col = np.median(overscan, axis=1)
        data -= overscan_col[:,np.newaxis]

        return data[self.regions["data"]]

    def zero_correct_frame(self, data, zero):
        """ Overscan subtract then zero subtract """

        d = self.overscan_subtract(data)
        d -= zero
        return d

class CCDRegion(list):

    def __init__(self, ccd, *slices):
        """ Represents a region / subset of a CCD detector

            Parameters
            ----------
            ccd : CCD
                The parent CCD object
            slices : tuple
                A tuple of slice objects which define the sub-region by
                slicing along each axis of the CCD.
        """
        self.ccd = ccd
        super(CCDRegion, self).__init__(*slices)

    @property
    def shape(self):
        ccd_shape = self.ccd.shape

        shp = []
        for ii,ax in enumerate(self):
            a, b = ax.start, ax.stop

            if a is not None and b is not None:
                sz = b-a
            elif a is None and b is not None:
                sz = ccd_shape[ii] - abs(b)
            elif a is not None and b is None:
                if a < 0:
                    sz = abs(a)
                else:
                    sz = ccd_shape[ii] - a
            elif a is None and b is None:
                sz = ccd_shape[ii]
            shp.append(sz)

        return tuple(shp)

class CCDFrame(object):
    pass