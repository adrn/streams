# coding: utf-8

""" A script for making figures for the Spitzer proposal """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import cPickle as pickle

# Third-party
from astropy.io import ascii
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc_context, rcParams
from matplotlib.patches import Rectangle
from matplotlib import cm

from streams.util import project_root
from streams.coordinates import ra_dec_dist_to_xyz
from streams.observation import apparent_magnitude
from streams.observation.gaia import parallax_error, proper_motion_error, \
                                     add_uncertainties_to_particles
from streams.observation.rrlyrae import rrl_M_V, rrl_V_minus_I

from streams.inference import relative_normalized_coordinates
from streams.inference.lm10 import timestep
from streams.potential import LawMajewski2010
from streams.potential.lm10 import true_params
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.io.lm10 import particle_table, particles_today, satellite_today, time
from streams.io.catalogs import read_quest
from streams.plot import sgr_kde

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('text', usetex=True)
matplotlib.rc('axes', edgecolor='#444444', labelsize=20)
matplotlib.rc('lines', markeredgewidth=0)
matplotlib.rc('font', family='sans-serif')
rcParams['font.sans-serif'] = 'helvetica'

plot_path = "plots/spitzer_proposal/"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

stripe82_catalog = """ra dec dist
14.616653 -0.75063 21.78
23.470286 -0.75074 20.02
25.920219 0.507809 18.37
28.105067 -0.28684 10.56
27.339723 -0.222394 19.37
28.709058 0.250204 7.59
29.908223 1.222851 18.35
27.37402 0.155047 19.87
25.46789 -0.89094 8.5
33.55261 0.660465 12.87
30.691522 -0.000466 20.11
30.13114 0.994329 20.66
32.419304 -0.036812 18.26
32.717291 0.631938 16.35
37.29115 1.07193 17.18
35.274198 -0.647943 12.27
40.612944 1.225526 7.29
40.508969 0.665032 20.11
44.338191 -0.220803 17.3
46.057459 -1.220635 18.74
47.609341 0.461527 9.4
52.601027 1.209482 9.24
51.883696 0.064283 17.62
53.635031 0.026812 7.44
53.80621 1.068064 10.51
56.963075 -0.291889 11.96
59.037319 -0.08628 7.19
58.036886 0.999678 8.18"""

quest_catalog = """ra dec dist
61.68371 -0.37483 6.91830970919
193.0365 -0.49203 9.6
193.18596 -0.42011 9.4
212.62708 -0.84478 20.3235701094
213.93104 -0.10356 19.3196831702
217.03742 -0.19667 15.7036280433
221.83508 -0.01711 28.0543363795
222.41508 -0.49544 23.0144181741
224.36254 -1.97 29.7851642943
226.42392 -1.65917 23.9883291902
227.47962 -1.54928 19.7696964011
227.75479 -1.63183 25.1188643151
227.86504 -1.98328 24.3220400907
228.98854 -0.11467 25.003453617
228.99154 -0.18975 23.7684028662
228.99296 -0.09647 26.6685866452
229.02412 -0.18675 25.2348077248
229.03967 -0.27306 25.2348077248
229.05329 -0.16742 24.3220400907
229.2045 -0.12967 26.0615355
229.43821 -0.64547 20.8929613085
229.86596 -0.64133 24.7742205763
230.34567 -0.09186 25.8226019063
230.82762 -0.92239 16.9824365246
231.78846 -0.34567 27.6694164541
231.98 -2.13494 25.7039578277
232.16571 -1.32111 26.0615355
232.91196 -0.21981 29.9226463661
233.04121 -1.3425 28.7078058202
234.94462 -0.62931 22.5943577022
235.344 -0.49136 21.2813904598
235.46812 -1.24069 22.3872113857
235.47917 -0.76497 24.3220400907
235.97929 -1.481 28.8403150313
236.5215 -0.4465 20.9893988362
237.11308 -1.31583 28.4446110745
237.80871 -0.97414 23.2273679636
239.08779 -0.10567 19.5884467351
239.15138 -2.19614 29.5120922667
240.39804 -1.23289 25.7039578277
240.74442 -0.23414 26.6685866452
242.22921 -0.26269 24.4343055269
242.31571 -2.01094 29.9226463661
242.33487 -1.00428 24.4343055269
245.36733 -1.01153 25.7039578277"""

def gaia_spitzer_errors():
    """ Visualize the observational errors from Gaia and Spitzer, along with
        dispersion and distance scale of Sgr and Orphan. 
    """
    
    rcparams = {'lines.linestyle' : '-', 
                'lines.linewidth' : 1.,
                'lines.color' : 'k',
                'lines.marker' : None,
                'axes.edgecolor' : '#444444',
                'axes.facecolor' : '#ffffff'}
    
    sgr_color = '#67A9CF'
    orp_color = '#EF8A62'
    
    with rc_context(rc=rcparams):
        fig,axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
        
        # vertical lines for dwarf satellites
        for name,dist in [('Sagittarius', 24), ('Ursa Minor, Bootes', 65), 
                          ('Sculptor', 80), ('Carina', 100)]:
            axes[0].axvline(dist, zorder=-1, alpha=0.4, linestyle='--')
            axes[1].axvline(dist, zorder=-1, alpha=0.4, linestyle='--')
        
        # label the vertical lines
        axes[1].text(16., 0.1, 'Sgr-', alpha=0.6)
        axes[1].text(27., 0.06, 'UMi, Boo-', alpha=0.6)
        axes[1].text(57., 0.03, 'Scl-', alpha=0.6)
        axes[1].text(68.5, 0.015, 'Car-', alpha=0.6)
        
        # Distance from 1kpc to ~100kpc
        D = np.logspace(0., 2., 50)*u.kpc
        
        # Sample metallicities from: http://arxiv.org/pdf/1211.7073v1.pdf
        fe_hs = np.random.normal(-1.67, 0.3, size=50)
        fe_hs = np.append(fe_hs, np.random.normal(-2.33, 0.3, size=len(fe_hs)//5))
        
        for fe_h in fe_hs:
            # Johnson/Cousins (V - I_C) color for RR Lyrae at *minimum*
            # Guldenschuh et al. (2005 PASP 117, 721), pg. 725
            rrl_V_minus_I = np.random.normal(0.579, 0.006)
            
            # Compute the apparent magnitude as a function of distance
            M_V = rrl_M_V(fe_h=fe_h)[0]
            m_V = apparent_magnitude(M_V, D)
            
            # Distance error
            dp = parallax_error(m_V, rrl_V_minus_I).arcsecond
            dD = D.to(u.pc).value**2 * dp * u.pc
            
            # Velocity error
            dpm = proper_motion_error(m_V, rrl_V_minus_I)
            dVtan = (dpm*D).to(u.km*u.radian/u.s).value
        
            # Plot Gaia distance errors
            axes[0].loglog(D.kiloparsec, (dD/D).decompose(), color='k', alpha=0.1)
                
            # Plot tangential velocity errors
            axes[1].loglog(D.kiloparsec, dVtan, color='k', alpha=0.1)
    
        # Add spitzer 2% line to distance plot
        axes[0].axhline(0.02, linestyle='-', linewidth=3, color='k', alpha=0.75)
    
    # Now add rectangles for Sgr, Orphan
    sgr_d = Rectangle((10., 0.15), 60., 0.15, 
                      color=sgr_color, alpha=1., label='Sgr stream width')
    axes[0].add_patch(sgr_d)
    
    # From fig. 3 in http://mnras.oxfordjournals.org/content/389/3/1391.full.pdf+html
    orp_d = Rectangle((10., 0.03), 35., 0.03,
                      color=orp_color, alpha=1., label='Orp stream width')
    axes[0].add_patch(orp_d)
    
    # Dispersion from Majewski 2004: 10 km/s
    sgr_v = Rectangle((10., 10), 60., 1., color=sgr_color, alpha=1.,
                      label='Sgr stream dispersion')
    axes[1].add_patch(sgr_v)
    
    orp_v = Rectangle((10., 8.), 35., 1., color=orp_color, alpha=1.,
                      label='Orp stream dispersion')
    axes[1].add_patch(orp_v)
    
    axes[0].set_ylim(0.003, 10)
    axes[0].set_xlim(1, 105)
    
    axes[1].set_ylim(0.01, 100)
    #axes[1].set_xlim(10, 100)
    
    axes[0].set_xlabel("Distance [kpc]")
    axes[0].set_ylabel("Frac. distance error $\sigma_D/D$")
    axes[1].set_ylabel("$v_{tan}$ error [km/s]")
    axes[1].set_xlabel("Distance [kpc]")
    
    axes[0].set_xticklabels(["1", "10", "100"])
    
    axes[0].set_yticklabels(["{:g}".format(yt) for yt in axes[0].get_yticks()])
    axes[1].set_yticklabels(["{:g}".format(yt) for yt in axes[1].get_yticks()])
    
    # add Gaia and Spitzer text to plots
    axes[0].text(4., 0.12, 'Gaia', fontsize=16, rotation=45)
    axes[0].text(4., 0.013, 'Spitzer', fontsize=16, color="k", alpha=0.75)
    axes[1].text(4., 0.23, 'Gaia + Spitzer', fontsize=16, rotation=45)
    
    # add legends
    axes[0].legend(loc='upper left', fancybox=True)
    axes[1].legend(loc='upper left', fancybox=True)
    
    axes[0].yaxis.tick_left()
    axes[1].yaxis.tick_left()
    
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "gaia.pdf"))

def sgr():
    """ Top-down plot of Sgr particles, with selected stars and then 
        re-observed
    """
    
    fig,axes = plt.subplots(1, 2, figsize=(14,6.5), sharex=True, sharey=True)
    
    # read in all particles as a table
    pdata = particle_table(N=0, expr="(Pcol<7) & (abs(Lmflag)==1)")
    
    extent = {'x':(-90,55), 
              'y':(-58,68)}
    Z = sgr_kde(pdata, extent=extent)
    axes[1].imshow(Z**0.5, interpolation="nearest", 
              extent=extent['x']+extent['y'],
              cmap=cm.Blues, aspect=1)
    
    np.random.seed(42)
    particles = particles_today(N=100, expr="(Pcol<7) & (Pcol>0) & (abs(Lmflag)==1)")   
    
    # read in Stripe 82 RR Lyrae
    s82 = ascii.read(stripe82_catalog)
    stripe82_xyz = ra_dec_dist_to_xyz(s82['ra']*u.deg, s82['dec']*u.deg, 
                                      s82['dist']*u.kpc)
    
    catalina = ascii.read(os.path.join(project_root, "data/spitzer_sgr_sample.txt"))
    catalina = catalina[catalina['dist'] < 35.]
    catalina_xyz = ra_dec_dist_to_xyz(catalina['ra']*u.deg, 
                                      catalina['dec']*u.deg, 
                                      catalina['dist']*u.kpc)
    
    quest = ascii.read(quest_catalog)
    quest = quest[quest['dist'] < 35]
    quest = quest[quest['dist'] > 10]
    quest_xyz = ra_dec_dist_to_xyz(quest['ra']*u.deg, 
                                   quest['dec']*u.deg, 
                                   quest['dist']*u.kpc)
    
    rcparams = {'lines.linestyle' : 'none', 
                'lines.color' : 'k',
                'lines.marker' : 'o'}
    
    with rc_context(rc=rcparams): 
        axes[0].plot(pdata['x'], pdata['z'],
                     marker='.', alpha=0.1, ms=5)
        
        axes[1].plot(stripe82_xyz[:,0], stripe82_xyz[:,2], marker='.', 
                     color='#111111', alpha=0.85, markersize=12, 
                     label="Stripe 82", markeredgewidth=0)
        
        axes[1].plot(catalina_xyz[:,0], catalina_xyz[:,2], marker='^', 
                     color='#111111', alpha=0.85, markersize=7, 
                     label="Catalina", markeredgewidth=0)
        
        axes[1].plot(quest_xyz[:,0], quest_xyz[:,2], marker='s', 
                     color='#111111', alpha=0.85, markersize=6, 
                     label="QUEST", markeredgewidth=0)
        
        # add solar symbol
        axes[0].text(-8., 0., s=r"$\odot$")
        axes[1].text(-8., 0., s=r"$\odot$")

        axes[0].set_xlabel("$X_{GC}$ [kpc]")
        axes[1].set_xlabel("$X_{GC}$ [kpc]")
        axes[0].set_ylabel("$Z_{GC}$ [kpc]")
    
    axes[1].legend(loc='upper left')
    plt.tight_layout()
    
    axes[0].set_xlim(extent['x'])
    axes[0].set_ylim(extent['y'])
    
    # turn off right, left ticks respectively
    axes[0].yaxis.tick_left()
    axes[1].yaxis.tick_right()
    
    # turn off top ticks
    axes[0].xaxis.tick_bottom()
    axes[1].xaxis.tick_bottom()
    
    fig.subplots_adjust(wspace=0.)
    fig.savefig(os.path.join(plot_path, "lm10.pdf"))

def bootstrapped_parameters_v1():
    data_file = os.path.join(project_root, "plots", "hotfoot", 
                             "SMASH", "all_best_parameters.pickle")
    
    with open(data_file) as f:
        data = pickle.load(f)
    
    fig,axes = plt.subplots(1,3,figsize=(16,6))

    y_param = 'v_halo'
    x_params = ['q1', 'qz', 'phi']
    
    lims = dict(q1=(1.3,1.45), qz=(1.3,1.45), v_halo=(115,130), phi=(90,105))
    
    for ii,x_param in enumerate(x_params):
        ydata = data[y_param]
        xdata = data[x_param]
        
        y_true = true_params[y_param]
        x_true = true_params[x_param]
        
        if y_param == 'v_halo':
            ydata = (ydata*u.kpc/u.Myr).to(u.km/u.s).value
            y_true = y_true.to(u.km/u.s).value
            
        if x_param == 'phi':
            xdata = (xdata*u.radian).to(u.degree).value
            x_true = x_true.to(u.degree).value
            
        axes[ii].axhline(y_true, linewidth=2, color='#2B8CBE', alpha=0.6)
        axes[ii].axvline(x_true, linewidth=2, color='#2B8CBE', alpha=0.6)
        axes[ii].plot(xdata, ydata, marker='o', alpha=0.75, linestyle='none')
        axes[ii].set_xlim(lims[x_param])
        axes[ii].set_ylim(lims[y_param])
        
        # turn off top ticks
        axes[ii].xaxis.tick_bottom()
    
    axes[0].set_ylabel(r"$v_{\rm halo}$", 
                       fontsize=26, rotation='horizontal')
    fig.text(0.015, 0.48, "[km/s]", fontsize=16)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    
    axes[0].set_xlabel(r"$q_1$", fontsize=26)
    axes[1].set_xlabel(r"$q_z$", fontsize=26)
    axes[2].set_xlabel(r"$\phi$", fontsize=26)
    fig.text(0.85, 0.05, "[deg]", fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.04)
    #plt.show()
    fig.savefig(os.path.join(plot_path, "bootstrap.pdf"))

if __name__ == '__main__':
    #gaia_spitzer_errors()
    #sgr()
    bootstrapped_parameters_v1()