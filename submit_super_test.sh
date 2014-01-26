#!/bin/sh

# Directives
#PBS -N infer_potential
#PBS -W group_list=yetiastro
#PBS -l nodes=16:ppn=16,walltime=16:00:00,mem=256gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

# print date and time to file
date

#Command to execute Python program
mpiexec -n 256 /vega/astro/users/amp2217/yt-x86_64/bin/python /vega/astro/users/amp2217/projects/streams/scripts/super_test.py --mpi -v -o --machine=yeti

date

#End of script