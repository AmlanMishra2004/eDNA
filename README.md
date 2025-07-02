# Summary
Identifies DNA sequences's species using a CNN.
Based off Jon Donnely and Rose Gurung's code, as well as functions from ETH Zurich's research paper "Applying convolutional neural networks to speed up environmental DNA annotation in a highly diverse ecosystem".

## Important files/folders

### saved_ppn_models/best_ppn_weights_only_1892566_8_-1.pth
- Our saved model weights for the protopnet at latent weight 0.7

### backbone_1-layer_95.4.pt
- Our saved model weights for our backbone CNN classifier

### protopnet/main.py
- This is the file that trains and evaluates the ProtoPNet, with the ability to modify different hyperparameters and get results averaged over multiple trials. Uses ppnet.py (the implementation of the ProtoPNet) and train_and_test.py (which trains and evaluates the network).

### evaluate_model.py
- This is the file that trains the base/backbone CNN classifier. It can explore any number of architectures with grid search using the VariableCNN class. This file reads in training data from ./datasets, creates the dataset object using dataset.py, gets one or more models from models.py, trains the model(s), and then evaluates the model(s). This file also holds the code that trains and evaluates the baseline models.
  
### /datasets (to download)
- 6GB Download link: https://drive.google.com/file/d/1I9E3uRZBdTFRBPXms59xjCR5bZJRPqMY/view?usp=sharing
- Contains the csv files that are used as training and testing data. Note that some files use a semicolon separator and others use a comma separator.
- Files starting with ft_N are feature tables for k-mers of length N

### dataset.py
- Contains the dataset object, which defines how the data goes through online augmentation.

### models.py
- Holds the definitions of a number of PyTorch models.
- Our backbone was the Small_Best_Updated class.

### schedule_job.slurm
- A file for creating a job for training or evaluation on a slurm computing cluster.

### utils.py
- Holds functions that are used by dataset.py and evaluate_model.py (for the backbone).
