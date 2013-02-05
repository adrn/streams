# coding: utf-8
"""
    Test the ...
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import pytest
import numpy as np
import astropy.units as u

from streams.simulation import *
from streams.potential import *

# API?:
disk_potential = MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
bulge_potential = HernquistPotential(M=3.4E10*u.solMass, c=0.7)
halo_potential = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value, q1=1.38, q2=1.0, qz=1.36, phi=1.692969, c=12.)
potential = disk_potential + bulge_potential + halo_potential
#potential = disk_potential

simulation = ParticleSimulation(potential=potential)

for ii in range(1000):
    r = np.sqrt(np.random.uniform()*100.)
    phi = 2.*np.pi*np.random.uniform()
    z = np.random.uniform(-0.5, 0.5)
    x = r*np.cos(phi)
    y = r*np.sin(phi)

    mag_v = (220.*u.km/u.s).to(u.kpc/u.Myr).value
    vx = -mag_v * sin(phi) # + (np.random.uniform()-0.5)*(dispersion/100.0/1.41),
    vy = mag_v * cos(phi) # + (np.random.uniform()-0.5)*(dispersion/100.0/1.41);
    vz = (np.random.uniform(-20., 20.)*u.km/u.s).to(u.kpc/u.Myr).value

    print (x,y,z)
    print (vx,vy,vz)
    p = Particle(position=(x, y, z), # kpc
                 velocity=(vx, vy, vz), # kpc/Myr
                 mass=1.) # M_sol
    simulation.add_particle(p)

assert (simulation.particles[0].position == simulation._particle_pos_array[0]).all()
assert (simulation.particles[0].velocity == simulation._particle_vel_array[0]).all()
assert (simulation.particles[11].position == simulation._particle_pos_array[11]).all()
assert (simulation.particles[11].velocity == simulation._particle_vel_array[11]).all()

ts, xs, vs = simulation.run(t1=0., t2=1000., dt=1.0)

num = 500

import matplotlib.pyplot as plt
import time

plt.clf()
fig, axes = plt.subplots(2,2, figsize=(12,12), sharex=True, sharey=True)
axes[0,1].set_visible(False)
axes[0,0].set_xlim(-50, 50)
axes[0,0].set_ylim(-50, 50)
fig.subplots_adjust(hspace=0.0, wspace=0.0)

circlesXY = axes[0,0].scatter(xs[0,:num,0], xs[0,:num,1], marker='o', s=2., c='k', alpha=0.5)
circlesXZ = axes[1,0].scatter(xs[0,:num,0], xs[0,:num,2], marker='o', s=2., c='k', alpha=0.5)
circlesYZ = axes[1,1].scatter(xs[0,:num,1], xs[0,:num,2], marker='o', s=2., c='k', alpha=0.5)

#print(xs[ii, :num, :2].shape)
#print(np.vstack((xs[ii, :num, 0], xs[ii, :num, 2])).T.shape)
#sys.exit(0)

for ii,t in enumerate(ts):
    circlesXY.set_offsets(xs[ii, :num, :2])
    circlesXZ.set_offsets(np.vstack((xs[ii, :num, 0], xs[ii, :num, 2])).T)
    circlesYZ.set_offsets(xs[ii, :num, 1:])

    #circles.set_facecolors(colors)
    plt.draw()
    time.sleep(0.02)

"""
plt.figure(figsize=(12,12))
circles = plt.scatter(xs[0,:num,0], xs[0,:num,1], marker='o', s=10., c='k', alpha=0.5)
plt.xlim(-40, 40)
plt.ylim(-40, 40)

for ii,t in enumerate(ts):
    circles.set_offsets(xs[ii, :num, :2])
    #circles.set_facecolors(colors)
    plt.draw()
    time.sleep(0.02)

#plt.show()

#simulation.run(t1=0., t2=1000., dt=0.1)
# animation = simulation.animate() # Make snapshots of all timesteps, and make an animation

#ax = simulation.plot(t=1000.)
#plt.show()

"""

def test_forward_backward_integration():
    """ Here I want to take the initial conditions from SGR_CEN, integrate forwards and
        compare to instead taking the final data and integrating backwards
    """

    # Read in Kathryn's simulated data
    sgrdata = SGRData(num_stars=1)
    x0,y0,z0 = sgrdata.satellite_center["x"][0],sgrdata.satellite_center["y"][0],sgrdata.satellite_center["z"][0]
    v_x0,v_y0,v_z0 = sgrdata.satellite_center["vx"][0],sgrdata.satellite_center["vy"][0],sgrdata.satellite_center["vz"][0]

    xf,yf,zf = sgrdata.satellite_center["x"][-1],sgrdata.satellite_center["y"][-1],sgrdata.satellite_center["z"][-1]
    v_xf,v_yf,v_zf = sgrdata.satellite_center["vx"][-1],sgrdata.satellite_center["vy"][-1],sgrdata.satellite_center["vz"][-1]

    t1, t2 = (min(sgrdata.satellite_center["t"]), max(sgrdata.satellite_center["t"]))
    dt = sgrdata.satellite_center["dt"]

    # Define potential as 3-component, bulge-disk-halo model
    disk_potential = MiyamotoNagaiPotential(M=1E11*u.solMass, a=6.5, b=0.26)
    bulge_potential = HernquistPotential(M=3.4E10*u.solMass, c=0.7)
    halo_potential = LogarithmicPotentialLJ(v_halo=(121.858*u.km/u.s).to(u.kpc/u.Myr).value, q1=1.38, q2=1.0, qz=1.36, phi=1.692969, c=12.)
    galaxy_potential = disk_potential + bulge_potential + halo_potential

    # Integrate initial conditions forward
    simulation = ParticleSimulation(potential=galaxy_potential)
    p = Particle(position=(x0,y0,z0), # kpc
                 velocity=(v_x0,v_y0,v_z0), # kpc/Myr
                 mass=1.) # M_sol
    simulation.add_particle(p)
    ts_forward, xs_forward, vs_forward = simulation.run(t1=t1, t2=t2, dt=sgrdata.satellite_center["dt"])

    # Integrate final conditions backward
    simulation = ParticleSimulation(potential=galaxy_potential)
    p = Particle(position=(xf,yf,zf), # kpc
                 velocity=(v_xf,v_yf,v_zf), # kpc/Myr
                 mass=1.) # M_sol
    simulation.add_particle(p)
    ts_back, xs_back, vs_back = simulation.run(t1=t2, t2=t1, dt=-sgrdata.satellite_center["dt"])

    plt.subplot(311)
    plt.plot(ts_forward, xs_forward[:,0,0], 'k-')
    plt.plot(ts_back, xs_back[:,0,0], 'b-')
    plt.plot(sgrdata.satellite_center["t"], sgrdata.satellite_center["x"], 'r--')

    plt.subplot(312)
    plt.plot(ts_forward, xs_forward[:,0,1], 'k-')
    plt.plot(ts_back, xs_back[:,0,1], 'b-')
    plt.plot(sgrdata.satellite_center["t"], sgrdata.satellite_center["y"], 'r--')

    plt.subplot(313)
    plt.plot(ts_forward, xs_forward[:,0,2], 'k-')
    plt.plot(ts_back, xs_back[:,0,2], 'b-')
    plt.plot(sgrdata.satellite_center["t"], sgrdata.satellite_center["z"], 'r--')

    plt.show()
