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
import astropy.coordinates as coord
from astropy.io import ascii
import matplotlib.pyplot as plt

from ... import usys
from ..leapfrog import LeapfrogIntegrator
from ..particle import *
from ...potential import *

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

def test_sgr_sun():
    from ...coordinates.frame import galactocentric, heliocentric
    from ...potential.lm10 import LawMajewski2010
    from ...io import SgrSimulation
    from ...dynamics import Particle
    from ...observation import apparent_magnitude

    potential = LawMajewski2010()
    simulation = SgrSimulation(mass="2.5e8")
    satellite = simulation.satellite()
    sun = Particle((-8.,0.,0.,0.,220.,0.),
                   frame=galactocentric,
                   units=(u.kpc,u.kpc,u.kpc,u.km/u.s,u.km/u.s,u.km/u.s))

    acc = np.zeros((2,3))
    pi = ParticleIntegrator((satellite,sun),
                            potential,
                            args=(2, acc))
    sgr_orbit,sun_orbit = pi.run(t1=0., t2=-2000., dt=-1.)

    D = np.sqrt(np.sum((sgr_orbit._X[:,:3] - sun_orbit._X[:,:3])**2, axis=-1))

    sgr_orbit_hel = sgr_orbit.to_frame(heliocentric)
    l,b = np.squeeze(sgr_orbit_hel._X[...,:2]).T
    gal = coord.Galactic(l*u.rad,b*u.rad)
    icrs = gal.icrs

    # plt.plot(icrs.ra.degree[0], icrs.dec.degree[0], marker='o')
    # plt.plot(icrs.ra.degree[:100], icrs.dec.degree[:100])
    # plt.show()

    fn = os.path.join(plot_path, "coords")
    with open(fn, 'w') as f:
        c = ["ra dec\n"]
        c = c + ["{:.3f} {:.3f}\n".format(r,d)
                  for r,d in zip(icrs.ra.degree, icrs.dec.degree)]
        f.writelines(c)

    fig,axes = plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)

    x,y,z = np.squeeze(sgr_orbit._X[...,:3].T)
    axes[0,0].plot(x,y,marker=None,linestyle='-')
    axes[1,0].plot(x,z,marker=None,linestyle='-')
    axes[1,1].plot(y,z,marker=None,linestyle='-')
    axes[0,1].set_visible(False)

    x,y,z = np.squeeze(sun_orbit._X[...,:3].T)
    axes[0,0].plot(x,y,marker=None,linestyle='-')
    axes[1,0].plot(x,z,marker=None,linestyle='-')
    axes[1,1].plot(y,z,marker=None,linestyle='-')

    axes[0,0].set_xlim(-60,60)
    axes[0,0].set_ylim(-60,60)

    fig,axes = plt.subplots(2,1,figsize=(12,8), sharex=True)
    V = np.squeeze(apparent_magnitude(-13.27, D*u.kpc))
    tbl = ascii.read(os.path.join(plot_path, "extinction.tbl"))
    AV = np.array(tbl['AV_SFD'].data)

    fig.suptitle("Sgr dwarf core heliocentric distance and V-band magnitude")
    axes[0].plot(sgr_orbit.t.value, D, lw=2.)
    axes[0].set_ylabel(r"$D_\odot$ [kpc]", rotation='horizontal', labelpad=35)
    axes[0].set_ylim(15,70)
    axes[1].plot(sgr_orbit.t.value, V, lw=2., label='uncorrected')
    axes[1].plot(sgr_orbit.t.value, V + AV, lw=2.,
                  label='w/ SFD extinction')
    axes[1].set_ylabel("$V$ [mag]", rotation='horizontal', labelpad=35)
    axes[1].set_xlabel("time [Myr]")
    axes[1].legend(loc='upper left', fontsize=12)
    axes[1].set_xlim(-1000., 0.)
    axes[1].set_ylim(2., 15)

    fig.subplots_adjust(hspace=0.02)
    fig.savefig(os.path.join(plot_path, "sgr_mag.pdf"))
