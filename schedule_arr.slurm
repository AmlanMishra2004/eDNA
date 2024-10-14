#!/bin/bash

# This file can be used to run main.py to search through different noise combinations or hyperparameter combinations for the ppnet.

#SBATCH --job-name=back_to_old_hparams  		# Job name
#SBATCH --cpus-per-task=5 		# Run on 5 cores per node
#SBATCH --nodes=1			# Run on 1 node
#SBATCH --partition=haswell		# Select partition
#SBATCH --mem=100gb			# Job memory request
#SBATCH --time=96:00:00		# Time limit, hrs:min:sec
#SBATCH --gres=gpu:1			# Request 1 gpu            
#SBATCH --output=slurm_outputs/out.%A_%a.log    # job id is %j, array index is %a
#SBATCH --array=1-9%6       # inclusive on both ranges. creates 500 jobs, do %15 at a time. creates a separate .out file for each

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

# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 baselines_noload.py --train_noise $train_noise --test_noise $test_noise

singularity run --nv ~/containers/pytorch-waggoner2.simg python3 main.py --arr_job_id $SLURM_ARRAY_JOB_ID --comb_num -1 --train_noise $train_noise --test_noise $test_noise # use if evaluating different noises, not hyperparamers
# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 main.py --arr_job_id $SLURM_ARRAY_JOB_ID --comb_num $SLURM_ARRAY_TASK_ID --train_noise 1 --test_noise 0 # use if evaluating different hyperparameters
# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 main.py --arr_job_id $SLURM_ARRAY_JOB_ID --comb_num $SLURM_ARRAY_TASK_ID --train_noise $train_noise --test_noise $test_noise # use if evaluating different noises and hyperparamers

# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 evaluate_model.py
# singularity run --nv ~/containers/pytorch-waggoner.simg python3 zurich_replica/code/exploration_full_clean.py
# singularity run --nv ~/containers/pytorch-waggoner.simg python3 testScript.py
# singularity run --nv $PYTORCH_CONT python3 zurich_replica/src/models/testScript.py
# singularity run --nv $PYTORCH_CONT home/sam/eDNA_env/bin/python3 zurich_replica/src/models/testScript.py
# singularity run --nv $PYTORCH_CONT python3 zurich_replica/src/models/exploration.py
