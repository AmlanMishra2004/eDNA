#!/bin/bash

mkdir -p slurm_outputs/$SLURM_ARRAY_JOB_ID

#SBATCH --job-name=python_job_array
#SBATCH --output=result-%A_%a.out
#SBATCH --array=0-10%5

module load singularity

srun python test.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID