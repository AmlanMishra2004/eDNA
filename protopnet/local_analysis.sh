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

# srun -u python3 local_analysis.py\
#                                 -gpuid 0\
#                                 -modeldir './saved_ppn_models'\
#                                 -model '1857326_0.9894.pth'\
#                                 -savedir ./local_results/test_local_seq_$IND\
#                                 -targetrow $IND \
#                                 -sequence 'ACGTCGTGTGTGTGTGTGTGGGT'\
#                                 -seqclass 1

# module load singularity

# singularity run --nv ~/containers/pytorch-waggoner2.simg python3 local_analysis.py\
#                                     -gpuid 0\
#                                     -modeldir './saved_ppn_models'\
#                                     -model '1857326_0.9894.pth'\
#                                     -savedir ./local_results/test_local_seq_$IND\
#                                     -targetrow $IND \
#                                     -sequence 'ACGTCGTGTGTGTGTGTGTGGGT'\
#                                     -seqclass 1

# source ../eDNA_env/bin/activate

for IND in 25 50 75 100 125 150 175 200
do
    srun -u python3 local_analysis.py\
                                    -gpuid 0\
                                    -modeldir './saved_ppn_models'\
                                    -model '1857326_0.9894.pth'\
                                    -savedir ./local_results/test_local_seq_$IND\
                                    -targetrow $IND \
                                    -sequence 'ACGTCGTGTGTGTGTGTGTGGGT'\
                                    -seqclass 1
    
done

