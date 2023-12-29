# Summary
Identifies DNA sequences's species using a CNN.
Based off the skeleton of Jon Donnely's code, as well as functions from ETH Zurich's research paper "Applying convolutional neural networks to speed up environmental DNA annotation in a highly diverse ecosystem".

## What's What
### ./datasets (to download)
- 6GB Download link: https://drive.google.com/file/d/1I9E3uRZBdTFRBPXms59xjCR5bZJRPqMY/view?usp=sharing
- Contains the csv files that are used as training and testing data. Note that some files use a semicolon separator and others use a comma separator. No need to modify.

### ./image_classifier (now removed from remote repo)
- Held the data for AutoKeras' search for the best-performing model. Removed 12/15/23 because its size was becoming much too large. No need to modify. 

### ./model_results
- Holds a number of text files. Each file shows the performance results of a single run of a grid search. This can mean multiple models or just one, one learning rate or multiple. 

### ./saved_models 
- Contains any models saved by AutoKeras while searching for an ideal model. No need to modify.

### dataset.py
- Contains the dataset object, which defines how the data goes through online augmentation.

### evaluate_model.py
- This is the heart of this project. This file reads in training data from ./datasets, creates the dataset object using dataset.py, gets one or more models from models.py, trains the model(s), and then evaluates the model(s). This is where you will start.

### kmer_baseline.ipynb
- This is a baseline that uses the k-mer approach and a naive bayes multinomial model.

### models.py
- Holds the definitions of any number of PyTorch models.

### schedule_job.slurm
- A file for creating a job and allocating resources on UMaine's supercomputer.

### utils.py
- Holds functions that are used by dataset.py and evaluate_model.py. These functions are defined here instead of in those files because it makes those files (which contain the essence of the logic) easier to read. Then, if someone wants to view implementational details, they may look at this file.

# Instructions
$ git clone https://github.com/SamWaggoner/eDNA.git  
Modify the settings in evaluate_model.py, deciding whether you want to run one of the models in models.py, or if you want to run autokeras or the baselines.
Modify the config dict in evaluate_model.py if you would like to run with different sequence lengths, noise rates, train/test split, and more.
$ python3 evaluate_model.py
