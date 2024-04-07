#!/bin/bash

#SBATCH --job-name=ppn_model_search  		# Job name
#SBATCH --cpus-per-task=5 		# Run on 5 cores per node
#SBATCH --nodes=1			# Run on 1 node
#SBATCH --partition=grtx		# Select partition
#SBATCH --mem=100gb			# Job memory request
#SBATCH --time=72:00:00		# Time limit, hrs:min:sec
#SBATCH --gres=gpu:1			# Request 1 gpu            
#SBATCH --output=slurm_outputs/out.%A_%a.log    # job id is %j, array index is %a
#SBATCH --array=1-25%13       # creates 500 jobs, do %15 at a time. creates a separate .out file for each

# Load versions of modules
module load singularity

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT: $SLURM_ARRAY_TASK_COUNT"
echo "SLURM_ARRAY_TASK_MAX: $SLURM_ARRAY_TASK_MAX"
echo "SLURM_ARRAY_TASK_MIN: $SLURM_ARRAY_TASK_MIN"

source ../eDNA_env/bin/activate
cd protopnet

echo "==========================================="
pwd; hostname; date;
echo "==========================================="

singularity run --nv ~/containers/pytorch-waggoner2.simg python3 main.py --arr_job_id $SLURM_ARRAY_JOB_ID --comb_num $SLURM_ARRAY_TASK_ID

# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 evaluate_model.py
# singularity run --nv ~/containers/pytorch-waggoner.simg python3 zurich_replica/code/exploration_full_clean.py
# singularity run --nv ~/containers/pytorch-waggoner.simg python3 testScript.py
# singularity run --nv $PYTORCH_CONT python3 zurich_replica/src/models/testScript.py
# singularity run --nv $PYTORCH_CONT home/sam/eDNA_env/bin/python3 zurich_replica/src/models/testScript.py
# singularity run --nv $PYTORCH_CONT python3 zurich_replica/src/models/exploration.py
