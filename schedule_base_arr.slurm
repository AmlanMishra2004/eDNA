#!/bin/bash

# For scheduling array jobs of the base CNN with evaluate_model.py

#SBATCH --job-name=ppn_model_search  		# Job name
#SBATCH --cpus-per-task=5 		# Run on 5 cores per node
#SBATCH --nodes=1			# Run on 1 node
#SBATCH --partition=grtx		# Select partition
#SBATCH --mem=100gb			# Job memory request
#SBATCH --time=72:00:00		# Time limit, hrs:min:sec
#SBATCH --gres=gpu:1			# Request 1 gpu            
#SBATCH --output=slurm_outputs/out.%A_%a.log    # job id is %j, array index is %a
#SBATCH --array=1-9%5       # inclusive on both ranges. creates 500 jobs, do %15 at a time. creates a separate .out file for each

# Load versions of modules
module load singularity

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT: $SLURM_ARRAY_TASK_COUNT"
echo "SLURM_ARRAY_TASK_MAX: $SLURM_ARRAY_TASK_MAX"
echo "SLURM_ARRAY_TASK_MIN: $SLURM_ARRAY_TASK_MIN"

source ../eDNA_env/bin/activate

echo "==========================================="
pwd; hostname; date;
echo "==========================================="

# Define the combinations of train_noise and test_noise
combinations=(
    "0 0"
    "0 1"
    "0 2"
    "1 0"
    "1 1"
    "1 2"
    "2 0"
    "2 1"
    "2 2"
)

# Get the combination for the current task
combination=${combinations[$SLURM_ARRAY_TASK_ID-1]}
train_noise=$(echo $combination | cut -d ' ' -f 1)
test_noise=$(echo $combination | cut -d ' ' -f 2)

echo "train_noise: $train_noise"
echo "test_noise: $test_noise"

singularity run --nv ~/containers/pytorch-waggoner2.simg python3 evaluate_model.py --train_noise $train_noise --test_noise $test_noise