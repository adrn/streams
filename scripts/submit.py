#!/hpc/astro/users/amp2217/yt-x86_64/bin/python
# coding: utf-8

""" Create and submit a job to the cluster given a streams config file. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Project
from streams.simulation.config import read

# Create logger
logger = logging.getLogger(__name__)

job_sh = """#!/bin/sh

# Directives
#PBS -N BootstrapLM10
#PBS -W group_list=hpcastro
#PBS -l nodes={nodes:d}:ppn=4,walltime={time},mem={memory}
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/hpc/astro/users/amp2217/jobs/output
#PBS -e localhost:/hpc/astro/users/amp2217/jobs/output

#Command to execute Python program
mpirun -n {walkers:d} /hpc/astro/users/amp2217/projects/streams/scripts/infer_potential.py -f /hpc/astro/users/amp2217/projects/streams/config/{config_file}

#End of script
"""

def main(config_file, walltime, memory):
    
    # Read in simulation parameters from config file
    config = read(config_file)
    
    d = config["walkers"] / 4
    if int(d) != d:
        raise ValueError()
        
    sh = job_sh.format(walkers=config["walkers"], 
                       nodes=config["walkers"]//4,
                       time=walltime,
                       memory=memory,
                       config_file=os.path.basename(config_file))
    
    print(sh)
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="")
    parser.add_argument("-f", "--file", dest="file", required=True, 
                    help="Path to the configuration file to run with.")
    parser.add_argument("-t", "--walltime", dest="time", default="36:00:00", 
                    help="Amount of time to request.")
    parser.add_argument("-m", "--memory", dest="memory", default="16gb", 
                    help="Amount of memory to request.")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)
    
    main(args.file, walltime=args.time, memory=args.memory)
    sys.exit(0)
 
