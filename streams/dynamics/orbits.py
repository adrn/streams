# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G
import triangle

from .core import _validate_quantity, DynamicalBase
from .particles import Particle

__all__ = ["Orbit"]

class Orbit(Particle):

    def __init__(self, t, coords, names, units=None, meta=dict()):
        """ Represents a particle orbit or collection of orbits.

            Parameters
            ----------
            t : quantity_like
                An array representing the time along the 0th axis of each
                coordinate, or the 1st axis along _X.
            coords : iterable
                Can either be an iterable (e.g., list or tuple) of Quantity
                objects, in which case their units are grabbed from the
                objects themselves, or an array_like object with the first
                axis as the separate coordinate dimensions, and the units
                parameter specifying the units along each dimension.
            names : iterable
                Names of each coordinate dimension. These end up being
                attributes of the object.
            units : iterable (optional)
                Must be specified if q is an array_like object, otherwise this
                is constructed from the Quantity objects in q.
            meta : dict (optional)
                Any additional metadata.
        """

        super(Orbit, self).__init__(coords, names, units=units, meta=meta)
        if not hasattr(t, "unit"):
            raise TypeError("'t' must be a quantity-like object with a .unit"
                            " attribute")

        self.t = t
        if self._X.shape[1] != self.t.shape[0]:
            raise ValueError("Shape of t ({}) should match 0th axis of each "             "coordinate ({})".format(self.t.shape[0], \
                                                      self._X.shape[1]))

    def plot(self, fig=None, labels=None, **kwargs):
        """ Make a corner plot showing all dimensions.

        """
        # TODO:

        if labels is None:
            labels = ["{0} [{1}]".format(n,uu) \
                        for n,uu in zip(self.names,self._repr_units)]

        if fig is not None:
            kwargs["fig"] = fig

        fig = triangle.corner(self._repr_X.T,
                              labels=labels,
                              plot_contours=False,
                              plot_datapoints=True,
                              **kwargs)

        return fig
