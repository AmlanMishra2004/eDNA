#!/bin/bash

#SBATCH --job-name=python_job_array_testing
#SBATCH --output=slurm_outputs/%A/testresult-%A_%a.out
#SBATCH --array=0-10%5

mkdir -p slurm_outputs/$SLURM_ARRAY_JOB_ID

module load singularity

srun python test.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID