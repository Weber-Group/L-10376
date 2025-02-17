#!/bin/bash

#SBATCH --partition=milano

# Request 5 hours of runtime: (always enter time in minutes)
#SBATCH --time=300:00

# Setting nodes and tasks per node: (Modify this if you're planning to use multiple nodes in parallel to run a job)
#SBATCH --nodes=4
#SBATCH --tasks-per-node=16

# Here's where the job name is defined (what shows up in the queue)
#SBATCH -J mask93

#Here's where the memory is defined
#SBATCH --mem=0

#This creates an output file
#SBATCH -o run_93.out

# This will email me when the job is finished (Just put your email in the second line)
#SBATCH --mail-type=END
#SBATCH --mail-user=nathan_goff@brown.edu 

# This runs the job
mpirun python -u -m mpi4py.run B_statistics_nouv_py3.py
