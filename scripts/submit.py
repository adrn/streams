# coding: utf-8

""" Create and submit a job to the cluster given a streams config file. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
from subprocess import Popen, PIPE
import cStringIO as StringIO
import yaml

# Create logger
logger = logging.getLogger(__name__)

job_sh = """#!/bin/sh

# Directives
#PBS -N {name}
#PBS -W group_list={group:s}astro
#PBS -l nodes={nodes:d}:ppn=8,walltime={time},mem={memory}
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:{astro:s}/pbs_output
#PBS -e localhost:{astro:s}/pbs_output

# print date and time to file
date

#Command to execute Python program
mpiexec -n {mpi_threads:d} {astro:s}/yt-x86_64/bin/python {astro:s}/projects/streams/scripts/{script} -f {astro:s}/projects/streams/config/{config_file} -v

date

#End of script
"""

def main(config_file, mpi_threads=None, walltime=None, memory=None,
         job_name=None, astro=None):

    # Read in simulation parameters from config file
    with open(config_file) as f:
        config = yaml.load(f.read())

    if job_name is None:
        if config.has_key("name"):
            name = config["name"]
        else:
            name = "adrn"
    else:
        name = job_name

    if mpi_threads is None:
        mpi_threads = 999999
    mpi_threads = min(config.get("walkers"), 256, mpi_threads)
    d = mpi_threads / 8
    if int(d) != d:
        raise ValueError()

    group = astro.split("/")[1]
    if group == "vega":
        group = 'yeti'

    sh = job_sh.format(mpi_threads=mpi_threads,
                       nodes=mpi_threads//8,
                       time=walltime,
                       memory=memory,
                       config_file=os.path.basename(config_file),
                       name=name,
                       script=config["script"],
                       astro=astro,
                       group=group)

    yn = raw_input("About to submit the following job: \n\n{0}\n\n Is "
                   "this right? [y]/n: ".format(sh))

    if yn.strip().lower() == "y" or yn.strip() == "":
        p = Popen(['qsub -'], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
        stdout_data = p.communicate(input=sh)[0]
        print("\n\n")
        print("Job started: {0}".format(stdout_data.split(".")[0]))
    else:
        sys.exit(1)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-f", "--file", dest="file", required=True,
                    help="Path to the configuration file to run with.")
    parser.add_argument("--walltime", dest="time", default="12:00:00",
                    help="Amount of time to request.")
    parser.add_argument("--memory", dest="memory", default="32gb",
                    help="Amount of memory to request.")
    parser.add_argument("--name", dest="job_name", default=None,
                    help="The name of the job.")
    parser.add_argument("--threads", dest="mpi_threads", default=None,
                        type=int, help="The number of MPI threads.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    ASTRO = os.environ['ASTRO']
    filename = os.path.join(ASTRO, "projects/streams/", args.file)
    main(filename, mpi_threads=args.mpi_threads, walltime=args.time,
         memory=args.memory, job_name=args.job_name, astro=ASTRO)
    sys.exit(0)

