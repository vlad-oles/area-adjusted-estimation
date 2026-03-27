#!/bin/bash

#export MPICH_DBG_LEVEL=ALL
#export MPICH_TRACEFILE=/tmp/mpi_trace_%d.log
#export MPICH_ABORT_ON_ERROR=1

#SBATCH --job-name=area-adj
#SBATCH --partition=reg
#SBATCH --ntasks=10000
#SBATCH --nodes=250
#SBATCH --cpus-per-task=1
#SBATCH -t 168:00:00 # run time (hh:mm:ss)
#SBATCH -o %j_out # output and error file name (%j expands to jobID)

# module load gcc mpich

echo

echo "Preparing run:"
date

#srun --mpi=pmi2 -u python -m mpi4py simulate.py
mpirun python simulate.py

echo "Run complete."
date

