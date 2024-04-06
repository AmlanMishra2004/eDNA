#!/bin/bash
#SBATCH --job-name=python_job_array
#SBATCH --output=result-%A_%a.out
#SBATCH --array=0-99

srun python test.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID