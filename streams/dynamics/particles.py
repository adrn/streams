# coding: utf-8

""" Massive particles or collections of particles """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import copy

# Third-party
import numpy as np
import astropy.units as u
import triangle

# Project
from .. import usys
from .core import DynamicalBase
from ..coordinates.frame import ReferenceFrame

__all__ = ["Particle"]

class Particle(DynamicalBase):

    def __init__(self, coords, frame, units=None, meta=dict()):
        """ Represents a dynamical particle or collection of particles.
            Particles can have associated metadata, e.g., mass or for
            a Satellite, velocity dispersion.

            Parameters
            ----------
            coords : iterable
                Can either be an iterable (e.g., list or tuple) of Quantity
                objects, in which case their units are grabbed from the
                objects themselves, or an array_like object with the first
                axis as the separate coordinate dimensions, and the units
                parameter specifying the units along each dimension.
            frame : ReferenceFrame
                The reference frame that the particles are in.
            units : iterable (optional)
                Must be specified if q is an array_like object, otherwise this
                is constructed from the Quantity objects in q.
            meta : dict (optional)
                Any additional metadata.
        """

        super(Particle, self).__init__(coords, frame, units=units, meta=meta)

    @property
    def coords(self):
        return tuple([self[n] for n in self.frame.coord_names])

    def to_frame(self, frame):
        """ Transform coordinates and reference frame.

            TODO: With astropy 0.4 (or 1.0) this will need to be
                  seriously updated.

            Parameters
            ----------
            frame : ReferenceFrame
        """

        if self.frame == frame:
            return self.copy()

        # frame transformation happens with usys units
        new_X = self.frame.to(frame, self._X)
        p = Particle(new_X.T,
                     frame=frame,
                     units=frame.units,
                     meta=self.meta)
        return p.to_units(frame.repr_units)

    def decompose(self, units):
        """ Decompose each coordinate axis to the given unit system """

        q = [self[n].decompose(units) for n in self.frame.coord_names]
        return Particle(q, frame=self.frame, meta=self.meta)

    def to_units(self, *units):
        """ Convert each coordinate axis to corresponding unit in given list. """

        if len(units) == 1:
            units = units[0]

        if len(units) != self.ndim:
            raise ValueError("Must specify a unit for each dimension ({})."\
                             .format(self.ndim))

        q = [self[n].to(units[ii]) \
                for ii,n in enumerate(self.frame.coord_names)]
        return Particle(q, frame=self.frame, meta=self.meta)

    def observe(self, errors):
        """ Assuming the current Particle object is in heliocentric
            coordinates, "observe" the positions given the errors
            specified in the dictionary "errors". The error dictionary
            should have keys == particles.names.
        """

        if self.frame.name != "heliocentric":
            raise ValueError("Particle must be in heliocentric frame.")

        new_qs = []
        for name in self.frame.coord_names:
            new_q = np.random.normal(self[name].value,
                                     errors[name].to(self[name].unit).value)
            new_q = new_q * self[name].unit
            new_qs.append(new_q)

        meta = self.meta.copy()
        meta["errors"] = errors
        return Particle(new_qs, frame=self.frame, meta=meta)

    def plot(self, fig=None, labels=None, \
             plot_kwargs=dict(), hist_kwargs=dict(),
             **kwargs):
        """ Make a corner plot showing all dimensions. """

        if labels is None:
            args = zip(self.frame.coord_names, self._repr_units)
            labels = ["{0} [{1}]".format(n,uu) \
                        for n,uu in args]

        plot_kwargs["alpha"] = plot_kwargs.get("alpha", 0.75)

        fig = triangle.corner(self._repr_X,
                              labels=labels,
                              plot_contours=False,
                              plot_datapoints=True,
                              fig=fig,
                              plot_kwargs=plot_kwargs,
                              hist_kwargs=hist_kwargs,
                              **kwargs)

        return fig

class ObservedParticle(Particle):

    def __init__(self, coords, errors, frame, units=None, meta=dict()):
        """ Represents a dynamical particle or collection of particles.
            Particles can have associated metadata, e.g., mass or for
            a Satellite, velocity dispersion.

            Parameters
            ----------
            coords : iterable
                Can either be an iterable (e.g., list or tuple) of Quantity
                objects, in which case their units are grabbed from the
                objects themselves, or an array_like object with the first
                axis as the separate coordinate dimensions, and the units
                parameter specifying the units along each dimension.
            frame : ReferenceFrame
                The reference frame that the particles are in.
            units : iterable (optional)
                Must be specified if q is an array_like object, otherwise this
                is constructed from the Quantity objects in q.
            meta : dict (optional)
                Any additional metadata.
        """
