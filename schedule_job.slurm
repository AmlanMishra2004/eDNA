#!/bin/bash

#SBATCH --job-name=ppn_model_search  		# Job name
#SBATCH --cpus-per-task=5 		# Run on 5 cores per node
#SBATCH --nodes=1			# Run on 1 node
#SBATCH --ntasks=1 			# One process (shared-memory)
#SBATCH --partition=grtx		# Select partition
#SBATCH --mem=100gb			# Job memory request
#SBATCH --time=72:00:00		# Time limit, hrs:min:sec
#SBATCH --gres=gpu:1			# Request 1 gpu            
#not SBATCH --output=out.%j.log		# Standard output and error log
#SBATCH --output=slurm_outputs/out.%j_%a.log    # job id is %j, array index is %a
#SBATCH --array=0-9%9       # creates 500 jobs, do %15 at a time. creates a separate .out file for each

# Load versions of modules
module load singularity

echo "==========================================="
pwd; hostname; date;
echo "==========================================="

# Evidently bash scripts execute from the path from which you call the script,
#   not from the path of the script itself. So, if you are in the directory 
#   bash_scripts/, then this will not work. (You would have to change it to be
#   ../zurich_replica/src/... Instead, do: ./bash_scripts/schedule_job.sh

source ../eDNA_env/bin/activate
cd protopnet

singularity run --nv ~/containers/pytorch-waggoner2.simg python3 main.py --job_id $SLURM_JOB_ID --index $SLURM_ARRAY_TASK_ID
# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 evaluate_model.py
# singularity run --nv ~/containers/pytorch-waggoner.simg python3 zurich_replica/code/exploration_full_clean.py
# singularity run --nv ~/containers/pytorch-waggoner.simg python3 testScript.py
# singularity run --nv $PYTORCH_CONT python3 zurich_replica/src/models/testScript.py
# singularity run --nv $PYTORCH_CONT home/sam/eDNA_env/bin/python3 zurich_replica/src/models/testScript.py
# singularity run --nv $PYTORCH_CONT python3 zurich_replica/src/models/exploration.py
