# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from ... import usys
from ..leapfrog import LeapfrogIntegrator
from ...potential import *
from ...potential.pal5 import Palomar5

plot_path = "plots/tests/integrate/pretty"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_log_potential():
    potential = LogarithmicPotentialLJ(units=usys,
                                           v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr),
                                           q1=1.38,
                                           q2=1.0,
                                           qz=1.36,
                                           phi=1.692969*u.radian,
                                           R_halo=12.*u.kpc)

    initial_position = np.array([[14.0, 0.0, 0.],
                                 [0.0, 14.0, 0.],
                                 [0.0, 14.0, 14.0]]).T # kpc
    initial_velocity = (np.array([[0.0, 160., 0.],
                                  [-160., 0., 0.],
                                  [150., 0., 16.]])*u.km/u.s).to(u.kpc/u.Myr).value.T

    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    initial_position.T, initial_velocity.T)
    ts, xs, vs = integrator.run(t1=0., t2=20000., dt=1.)

    fig,ax = plt.subplots(1,1,figsize=(8.5,11))
    ax.plot(xs[:,0,0], xs[:,0,1], color='#333333', marker=None, alpha=0.25, lw=3.)
    ax.plot(xs[:,1,0], xs[:,1,1], color='#0571B0', marker=None, alpha=0.3, lw=3.)
    ax.plot(xs[:,2,0], xs[:,2,1], color='#CA0020', marker=None, alpha=0.3, lw=3.)
    ax.set_axis_bgcolor("#777777")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    #ax.set_xlim(-20,20)
    #ax.set_ylim(-25.88,25.88)
    fig.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    fig.savefig(os.path.join(plot_path,"logarithmic.pdf"))
