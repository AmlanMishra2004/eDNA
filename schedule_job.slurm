#!/bin/bash

#SBATCH --job-name=running_baselines  		# Job name
#SBATCH --cpus-per-task=20 		# Run on 5 cores per node
#SBATCH --nodes=1			# Run on 1 node
#SBATCH --partition=grtx		# Select partition
#SBATCH --mem=500gb			# Job memory request
#SBATCH --time=72:00:00		# Time limit, hrs:min:sec
#SBATCH --gres=gpu:1			# Request 1 gpu            
#SBATCH --output=out.%j.log		# Standard output and error log
#SBATCH --ntasks=1 			# One process (shared-memory)

# Load versions of modules
module load singularity

# Evidently bash scripts execute from the path from which you call the script,
#   not from the path of the script itself. So, if you are in the directory 
#   bash_scripts/, then this will not work. (You would have to change it to be
#   ../zurich_replica/src/... Instead, do: ./bash_scripts/schedule_job.sh

echo "SLURM_JOB_ID: $SLURM_JOB_ID"

source ../eDNA_env/bin/activate
# cd protopnet

echo "==========================================="
pwd; hostname; date;
echo "==========================================="

singularity run --nv ~/containers/pytorch-waggoner2.simg python3 evaluate_model.py
# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 main.py

# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 evaluate_model.py
# singularity run --nv ~/containers/pytorch-waggoner.simg python3 zurich_replica/code/exploration_full_clean.py
# singularity run --nv ~/containers/pytorch-waggoner.simg python3 testScript.py
# singularity run --nv $PYTORCH_CONT python3 zurich_replica/src/models/testScript.py
# singularity run --nv $PYTORCH_CONT home/sam/eDNA_env/bin/python3 zurich_replica/src/models/testScript.py
# singularity run --nv $PYTORCH_CONT python3 zurich_replica/src/models/exploration.py
