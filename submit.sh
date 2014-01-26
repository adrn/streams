#!/bin/sh

# Directives
#PBS -N infer_potential
#PBS -W group_list=yetiastro
#PBS -l nodes=8:ppn=16,walltime=24:00:00,mem=156gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

# print date and time to file
date

#Command to execute Python program
mpiexec -n 128 /vega/astro/users/amp2217/yt-x86_64/bin/python /vega/astro/users/amp2217/projects/streams/scripts/infer_potential.py --mpi -v -o --file=/vega/astro/users/amp2217/projects/streams/config/test.yml

date

#End of script