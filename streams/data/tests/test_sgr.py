# coding: utf-8
"""
    Test the helper classes for reading in Sgr data.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

from .. import LM10Snapshot, SgrCen, SgrSnapshot
from ..gaia import add_uncertainties_to_particles

class TestSgrCen(object):
    sgr_cen = SgrCen()
    
    def test_create(self):
        sgr_cen = self.sgr_cen
        
        assert sgr_cen["x"].units == u.kpc
        assert sgr_cen["y"].units == u.kpc
        assert sgr_cen["z"].units == u.kpc
        
        assert sgr_cen["vx"].units == u.kpc/u.Myr
        assert sgr_cen["vy"].units == u.kpc/u.Myr
        assert sgr_cen["vz"].units == u.kpc/u.Myr
        
        assert sgr_cen.xyz.unit == u.kpc
        assert sgr_cen.vxyz.unit == u.kpc/u.Myr
        
        assert sgr_cen.xyz.value.shape == (3,len(sgr_cen))
        assert sgr_cen.vxyz.value.shape == (3,len(sgr_cen))
        
        assert sgr_cen["x"].shape == sgr_cen["y"].shape
        assert sgr_cen["y"].shape == sgr_cen["z"].shape
        
        assert sgr_cen["vx"].shape == sgr_cen["vy"].shape
        assert sgr_cen["vy"].shape == sgr_cen["vz"].shape
    
    def test_orbit(self):
        sgr_cen = self.sgr_cen
        
        sgr_cen.as_orbit()

class TestSgrSnap(object):
    
    def test_create(self):
        sgr_snap = SgrSnapshot(N=100, 
                               expr="sqrt(x**2 + y**2 + z**2) < 30.")
        
        assert np.max(np.sqrt(sgr_snap["x"]**2+sgr_snap["y"]**2+sgr_snap["z"]**2)) < 30.
        assert len(sgr_snap) == 100
        
        sgr_snap = SgrSnapshot(N=100, 
                               expr="(sqrt(x**2 + y**2 + z**2) < 20.) & (tub > 0)")
        assert np.max(np.sqrt(sgr_snap["x"]**2+sgr_snap["y"]**2+sgr_snap["z"]**2)) < 20.
        assert np.all(sgr_snap["tub"] > 0)
        assert len(sgr_snap) == 100
        
        sgr_snap = SgrSnapshot(N=100, 
                               expr="(tub > 0)")
        assert np.all(sgr_snap["tub"] > 0)
        assert len(sgr_snap) == 100

    def test_uncertainties(self):
        sgr_snap = SgrSnapshot(N=100, 
                               expr="(tub > 0)")
        
        p = sgr_snap.as_particles()
        with_errors = sgr_snap.add_errors()
        p_we = with_errors.as_particles()
        
        fig,axes = p.plot_r(coord_names=["X","Y","Z"], 
                            subplots_kwargs=dict(figsize=(16,16)))
        p_we.plot_r(coord_names=["X","Y","Z"], 
                    axes=axes, scatter_kwargs={"c":"r"})
        
        fig.savefig("plots/tests/sgrsnap_uncertainties_position.png")
        
        fig,axes = p.plot_v(coord_names=["Vx","Vy","Vz"], 
                            subplots_kwargs=dict(figsize=(16,16)))
        p_we.plot_v(coord_names=["Vx","Vy","Vz"], 
                    axes=axes, scatter_kwargs={"c":"r"})
        
        fig.savefig("plots/tests/sgrsnap_uncertainties_velocity.png")
    
    def test_as_particles(self):
        sgr_snap = SgrSnapshot(N=100, 
                               expr="(tub > 0)")
        p = sgr_snap.as_particles()

class TestLM10Snap(object):
    
    def test_create(self):
        snap = LM10Snapshot(N=100, 
                            expr="(dist<30) & (Pcol > 0)")
        
        assert np.max(snap["dist"]) < 30.
        assert len(snap) == 100
        
        snap = LM10Snapshot(N=100, 
                               expr="(dist<20) & (Pcol > 0)")
        assert np.max(snap['dist']) < 20.
        assert np.all(snap["Pcol"] > 0)
        assert len(snap) == 100
        
        snap = LM10Snapshot(N=100, 
                               expr="(Pcol > 0)")
        assert np.all(snap["Pcol"] > 0)
        assert len(snap) == 100

    def test_uncertainties(self):
        snap = LM10Snapshot(N=100, 
                               expr="(Pcol > 0)")
        
        p = snap.as_particles()
        p_we = add_uncertainties_to_particles(p)
        
        fig,axes = p.plot_r(coord_names=["X","Y","Z"], 
                            subplots_kwargs=dict(figsize=(16,16)))
        p_we.plot_r(coord_names=["X","Y","Z"], 
                    axes=axes, scatter_kwargs={"c":"r"})
        
        fig.savefig("plots/tests/sgrsnap_uncertainties_position_lm10.png")
        
        fig,axes = p.plot_v(coord_names=["Vx","Vy","Vz"], 
                            subplots_kwargs=dict(figsize=(16,16)))
        p_we.plot_v(coord_names=["Vx","Vy","Vz"],
                    axes=axes, scatter_kwargs={"c":"r"})
        
        fig.savefig("plots/tests/sgrsnap_uncertainties_velocity_lm10.png")
    
    def test_as_particles(self):
        sgr_snap = LM10Snapshot(N=100, 
                               expr="(Pcol > 0)")
        p = sgr_snap.as_particles()