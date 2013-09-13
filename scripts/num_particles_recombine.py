# coding: utf-8

""" For responding to the referee for paper1 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io.misc import fnpickle, fnunpickle

# Project
from streams.observation.gaia import add_uncertainties_to_particles
from streams.io.sgr import mass_selector
from streams.potential.lm10 import true_params, LawMajewski2010
from streams.util import project_root
from streams.integrate.satellite_particles import SatelliteParticleIntegrator
from streams.inference import minimum_distance_matrix

# Create logger
logger = logging.getLogger(__name__)

# plt.clf()
# for ii in range(6):
# plt.subplot(2,3,ii+1)
# plt.hist(min_ps[...,ii], 50)
# plt.xlim(-3, 3)
# 
# plt.savefig(os.path.join(plot_path, "{0}.png".format(mass)))

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", 
                    default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
                    default=False, help="Be quiet! (default = False)")
    parser.add_argument("-s", "--seed", dest="seed", default=42, 
                    help="Seed for numpy's random number generator.")
    parser.add_argument("-N", dest="N", default=1000, type=int,
                    help="Number of particles")
    parser.add_argument("-D", dest="D_ps_lim", default=2., type=float,
                    help="Cutoff for phase-space distance")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", 
                    default=False, help="Overwrite pre-existing shite.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    N_particles = args.N
    Dps_lim = args.D_ps_lim
    
    sgr_path = os.path.join(project_root, "data", "simulation", "Sgr")
    plot_path = os.path.join(project_root, "plots", "tests", "num_combine")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    
    data_file = os.path.join(plot_path, "frac_Dps{0}.pickle".format(Dps_lim))
    
    if args.overwrite and os.path.exists(data_file):
        os.remove(data_file)
    
    masses = ['2.5e{0}'.format(xx) for xx in range(6, 10)]
    canonical_errors = {"proper_motion_error_percent" : 1.,
                        "distance_error_percent" : 2.,
                        "radial_velocity_error" : 10.*u.km/u.s}
    error_fracs = [0.0001, 0.01, 0.1, 0.5, 1., 2.]
                    
    if not os.path.exists(data_file):
        lm10 = LawMajewski2010()
        
        frac_recombined = []
        for mass in masses:
            logger.info("Starting mass {0}...".format(mass))
            particles_today, satellite_today, time = mass_selector(mass) 
                    
            np.random.seed(args.seed)
            true_particles = particles_today(N=N_particles)
            satellite = satellite_today()
            t1, t2 = time()
            
            frac_recombined_per_mass = []
            for ii,errors in enumerate(error_dicts):
                if ii > 0:
                    particles = add_uncertainties_to_particles(true_particles, **errors)
                else:
                    particles = true_particles
                
                logger.info("\t with errors: {0}...".format(errors))
                
                # integrate the orbits backwards, compute the minimum phase-space distance
                integrator = SatelliteParticleIntegrator(lm10, satellite, particles)
                s_orbit,p_orbits = integrator.run(t1=t1, t2=t2, dt=-1.)
                min_ps = minimum_distance_matrix(lm10, s_orbit, p_orbits)
                
                D_ps = np.sqrt(np.sum(min_ps**2, axis=-1))
                frac = np.sum(D_ps < Dps_lim) / N_particles
                #print(mass, ii, frac)
                frac_recombined_per_mass.append(frac)
            
            frac_recombined.append(frac_recombined_per_mass)
        
        frac_recombined = np.array(frac_recombined).T
        fnpickle(frac_recombined, data_file)
    
    frac_recombined = fnunpickle(data_file)    
    kwargs = dict(marker="o", linestyle="-", lw=1., alpha=0.95)
    
    plt.figure(figsize=(12,8))
    for ii in range(len(frac_recombined)):
        pp = kwargs.copy()
        if ii == 2:
            pp['lw'] = 2.
            pp['c'] = 'k'
        
        if jj == 0:
            plt.semilogx([float(m) for m in masses], frac_recombined[ii], 
                         label=error_dicts[ii]['distance_error_percent'], **pp)
        else:
            plt.semilogx([float(m) for m in masses], frac_recombined[ii], **pp)
    
    plt.legend(title='$\sigma_D/D$')
    plt.xticks([float(m) for m in masses], masses)
    plt.xlabel("Satellite mass [$M_\odot$]", fontsize=26)
    plt.ylabel("# of particles that recombine", fontsize=26)
    plt.tight_layout()
    plt.show()