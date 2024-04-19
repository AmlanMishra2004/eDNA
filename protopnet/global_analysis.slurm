#!/bin/bash

#SBATCH --job-name=local_analysis  # Job name
# not SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
# not SBATCH --mail-user=rose.gurung@maine.edu     # Where to send mail
#SBATCH --output=data_prep_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=24
#SBATCH --mem=12gb                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1

module load singularity

source ../../eDNA_env/bin/activate

echo "==========================================="
pwd; hostname; date;
echo "==========================================="

for IND in 0 17 35 52 69 87 104 121 139 156
do
    singularity run --nv ~/containers/pytorch-waggoner2.simg python3 global_analysis.py \
                                    -gpuid 0 \
                                    -modeldir './saved_ppn_models' \
                                    -model '1857326_0.9894.pth' \
                                    -prototypeind $IND

done
                                


