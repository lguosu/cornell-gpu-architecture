#!/bin/bash
#SBATCH -J gpu_test         # Job name
#SBATCH -o gpu_test.o%j     # Output and error file name
#SBATCH -N 1                # Total number of GPU nodes requested
#SBATCH -n 1                # Total cores needed for the job
#SBATCH -p rtx-dev          # Queue name
#SBATCH -t 00:05:00         # Run time (hh:mm:ss)
##SBATCH -A [account]       # Project number (uncomment to specify which one)

./bandwidthTest
