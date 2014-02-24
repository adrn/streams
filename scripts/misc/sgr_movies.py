# coding: utf-8

""" Make movies of satellite disruption from David's Sgr simulations """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import astropy.units as u
from astropy.constants import G
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm, animation
from matplotlib.patches import Rectangle, Ellipse, Circle

# Project
from streams.util import project_root

matplotlib.rc('axes', edgecolor='#333333', 
              linewidth=2.0)
matplotlib.rc('lines', markeredgewidth=0, 
              linestyle='none', 
              marker='.',
              markersize=6.)
matplotlib.rc('font', family='Source Sans Pro', weight='light')

plot_path = os.path.join(project_root, "plots/movies/")
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def _units_from_file(scfpar):
    """ Generate a unit system from an SCFPAR file. """

    with open(scfpar) as f:
        lines = f.readlines()
        length = float(lines[16].split()[0])
        mass = float(lines[17].split()[0])

    GG = G.decompose(bases=[u.kpc,u.M_sun,u.Myr]).value
    X = (GG / length**3 * mass)**-0.5

    length_unit = u.Unit("{0} kpc".format(length))
    mass_unit = u.Unit("{0} M_sun".format(mass))
    time_unit = u.Unit("{:08f} Myr".format(X))

    return dict(length=length_unit,
                mass=mass_unit,
                time=time_unit)

def main(logmass):
    top_path = "/vega/astro/users/dah2154/sgr_plummer"
    sub_path = "M2.5e+{:02d}/4.0Gyr/L1.0/".format(int(logmass))
    path = os.path.join(top_path, sub_path)
    save_file = os.path.join(plot_path, "2.5e{}.mp4".format(logmass))

    cen_filename = os.path.join(path, "SCFCEN")
    units = _units_from_file(os.path.join(path, "SCFPAR"))

    cen = np.loadtxt(cen_filename, skiprows=2, usecols=(2,4))
    cen_x,cen_z = (cen*units['length']).to(u.kpc).value.T

    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(xlim=(-100, 100), ylim=(-100, 100))
    line_p, = ax.plot([], [], color='#92c5de', alpha=0.35)

    def init():
        line_p.set_data([], [])
        return line_p,

    def animate(i):
        ii = i + 1
        print(ii)

        filename = os.path.join(path, "SNAP{:03d}".format(ii))
        # columns are m,x,y,z,vx,vy,vz
        data = np.loadtxt(filename, skiprows=1, usecols=(1,3))
        x,z = (data[:,:3]*units['length']).to(u.kpc).value.T
        #vxyz = (data[:,3:]*units['length']/units['time']).to(u.kpc).value
        line_p.set_data(x-cen_x[i], z-cen_z[i])
        return line_p,
    
    frames = 5 # 533
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=8, blit=True)
    # extra_args=['-vcodec','libx264'],
    anim.save("test.mp4", fps=20) 
    #          savefig_kwargs=dict(facecolor='#333333'))
    

if __name__ == "__main__":
    main(int(sys.argv[1]))
