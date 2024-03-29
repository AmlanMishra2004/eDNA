import os
# import shutil

import itertools
import torch
import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from tqdm import tqdm
from sklearn import metrics
import warnings

import argparse
# import re
import random
import pandas as pd
import torch.nn as nn
import numpy as np

# from helpers import makedir
from push import push_prototypes
import train_and_test as tnt
# from preprocess import mean, std, preprocess_input_function
import sys
from sklearn.model_selection import train_test_split

# adding the parent folder to the system path so we can import from it
sys.path.append('..')
import utils
import models
import ppnet as ppn
from dataset import Sequence_Data
from torch.utils.data import DataLoader


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
        # Question: do we need to flush()
    return logger, f.close


config = {
    # IUPAC ambiguity codes represent if we are unsure if a base is one of
    # several options. For example, 'M' means it is either 'A' or 'C'.
    'iupac': {'a':'t', 't':'a', 'c':'g', 'g':'c',
            'r':'y', 'y':'r', 'k':'m', 'm':'k', 
            'b':'v', 'd':'h', 'h':'d', 'v':'b',
            's':'w', 'w':'s', 'n':'n', 'z':'z'},
    'raw_data_path': '../datasets/v4_combined_reference_sequences.csv',
    'train_path': '../datasets/train.csv',
    'test_path': '../datasets/test.csv',
    'sep': ';',                       # separator character in the csv file
    'species_col': 'species_cat',     # name of column containing species
    'seq_col': 'seq',                 # name of column containing sequences

    # Logged information (plus below):
    'verbose': True,
    'seq_count_thresh': 2,            # ex. keep species with >1 sequences
    'trainRandomInsertions': [0,2],   # ex. between 0 and 2 per sequence
    'trainRandomDeletions': [0,2],    # ex. between 0 and 2 per sequence
    'trainMutationRate': 0.05,        # n*100% chance for a base to flip
    'oversample': True,               # whether or not to oversample train # POSSIBLY OVERRIDDEN IN ARCH SEARCH
    'encoding_mode': 'probability',   # 'probability' or 'random'
    'push_encoding_mode': 'probability',   # 'probability' or 'random'
    # Whether or not applying on raw unlabeled data or "clean" ref db data.
    'applying_on_raw_data': False,
    # Whether or not to augment the test set.
    'augment_test_data': True,
    'load_existing_train_test': True, # use the same train/test split as Zurich, already saved in two different csv files
    'train_batch_size': 156, # prev. 64. 1,2,3,4,5,6,10,12,13,15,20,26,30,39,52,60,65,78,130,156,195,260,390,=780
    'test_batch_size': 94, # 1, 2, 4, 47, 94, 188 NOT # 1,5,7,25,35,175
    'num_classes': 156
}
if config['applying_on_raw_data']:
    config['seq_target_length'] = 150
    config['addTagAndPrimer'] = True 
    config['addRevComplements'] = True
elif not config['applying_on_raw_data']:
    # Logged data:
    config['seq_target_length'] = 70        # 70 (prev. 71) or 60 or 64
    config['addTagAndPrimer'] = False
    config['addRevComplements'] = False
if config['augment_test_data']:
    config['testRandomInsertions'] = [1,1]
    config['testRandomDeletions'] = [1,1]
    config['testMutationRate'] = 0.02
elif not config['augment_test_data']:
    config['testRandomInsertions'] = [0,0]
    config['testRandomDeletions'] = [0,0]
    config['testMutationRate'] = 0
assert config['seq_target_length'] % 2 == 0, \
    "Error: sequence length must be even"


# allow the user to call the program with command line gpu argument
# ex. python3 main.py -gpuid=0,1,2,3
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

# model_dir = './ppn_saved_models/' + base_architecture + '/' + experiment_run + '/'
# makedir(model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'base_model.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

# log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
# seq_dir = os.path.join(model_dir, 'seq')
# makedir(seq_dir)
# weight_matrix_filename = 'output_weights'
# prototype_seq_filename_prefix = 'prototype-seq'
# prototype_self_act_filename_prefix = 'prototype-self-act'
# proto_bound_boxes_filename_prefix = 'bb'

# normalize = transforms.Normalize(mean=mean, std=std) # not used in this file or any other?

# Jon says: we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)

# Unused since the number of pushes and the push start and gap defines how many
# epochs it runs.
# early_stopper = utils.EarlyStopping(
#     patience=7,
#     min_pct_improvement=1,#1,#3, # previously 20 epochs, 0.1% (for backbone)
#     verbose=False
# )

log = print
random.seed(36)
warnings.filterwarnings('ignore', 'y_pred contains classes not in y_true')

# split data into train, validation, and test
train = pd.read_csv(config['train_path'], sep=',')
test = pd.read_csv(config['test_path'], sep=',')
val_portion = 0.2
X_train, X_val, y_train, y_val = train_test_split(
    train[config['seq_col']], # X
    train[config['species_col']], # y
    test_size=val_portion,
    random_state=42,
    stratify = train[config['species_col']]
)
# print(X_train.shape)
# print(y_train.shape)
# print(type(X_train))
# print(type(y_train))

train = X_train.to_frame().join(y_train.to_frame()) # USED TO BE train = pd.concat([X_train, y_train], axis=1)

# print("success")
# wait=input("pause")
# orig_train = pd.concat([[1,2,3],[3,2,3]], axis=1)

if config['oversample']:
    train = utils.oversample_underrepresented_species(
        train,
        config['species_col'],
        config['verbose']
    )
train_dataset = Sequence_Data(
    X=train[config['seq_col']],
    y=train[config['species_col']],
    insertions=config['trainRandomInsertions'],
    deletions=config['trainRandomDeletions'],
    mutation_rate=config['trainMutationRate'],
    encoding_mode=config['encoding_mode'],
    seq_len=config['seq_target_length']
)
val_dataset = Sequence_Data(
    X_val,
    y_val,
    insertions=[config['testRandomInsertions']],
    deletions=config['testRandomDeletions'],
    mutation_rate=config['testMutationRate'],
    encoding_mode=config['encoding_mode'],
    seq_len=config['seq_target_length']
)
test_dataset = Sequence_Data(
    test[config['seq_col']],
    test[config['species_col']],
    insertions=[config['testRandomInsertions']],
    deletions=config['testRandomDeletions'],
    mutation_rate=config['testMutationRate'],
    encoding_mode=config['encoding_mode'],
    seq_len=config['seq_target_length']
)
trainloader = DataLoader(
    train_dataset, 
    batch_size=config['train_batch_size'],
    shuffle=True
)
valloader = DataLoader(
    val_dataset,
    batch_size=config['test_batch_size'],
    shuffle=False
)
testloader = DataLoader(
    test_dataset,
    batch_size=config['test_batch_size'],
    shuffle=False
)
# Push data should be raw data. This is raw data except that sequences for
# species with <2 sequences are removed. Used in training.
push_dataset = Sequence_Data(
    X_train,
    y_train,
    insertions=[0,0],
    deletions=[0,0],
    mutation_rate=0,
    encoding_mode=config['push_encoding_mode'],
    seq_len=config['seq_target_length']
)
pushloader = DataLoader(
    push_dataset,
    batch_size=config['train_batch_size'],
    shuffle=False
)

# model_path = "../best_model_20240103_210217.pt" # updated large best
# model_path = "../best_model_20240111_060457.pt" # small best with pool=3, stride=1
# model_path = "../best_model_20240126_120130.pt" # small best with pool=2,stride=2
model_path = "../small_best_updated_backbone.pt" # small best with pool=2,stride=2
# large_best = models.Large_Best()
backbone = models.Small_Best_Updated()
backbone.load_state_dict(torch.load(model_path))
backbone.linear_layer = nn.Identity() # remove the linear layer

# begin hyperparameter search
# this is the number of times you want to repeat either the
# grid search below, or the random search below.
for trial in range(1):

    print(f"\n\nTrial {trial+1}\n")
    # early_stopper.reset()

    print('training set size: {0}'.format(len(trainloader.dataset)))
    print('push set size: {0}'.format(len(pushloader.dataset)))
    print('test set size: {0}'.format(len(testloader.dataset)))
    print('train batch size: {0}'.format(config['train_batch_size']))
    print('test batch size: {0}'.format(config['test_batch_size']))

    num_latent_channels = 512
  
    """
    If you want to do GRID SEARCH, then set the number of trials above to 1
    and defined the lists of numbers to try below.

    If you want to do do RANDOM SEARCH, then set the number of trials above to
    however many combinations you want to try, and define the random
    distributions below.
    """
    # search 2/25/24
    # # These two are also hyperparameters. Feel free to add more values to try.
    # num_ptypes_per_class = [2] #random.randint(1, 3) # not set
    # ptype_length = [25] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    # hyperparameters = {    
    #     # comments after the line indicate jon's original settings
    #     # if the settings were not applicable, I write "not set".

    #     'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
    #     'latent_weight':            [0.5, 0.6, 0.7, 0.8, 0.9], #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
    #     'joint_weight_decay':       [0.065], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
    #     'gamma':                    [1], #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
    #     'warm_lr_step_size':        [1_000_000], # try 14? #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'coefs': [{ # weighting of different training losses
    #         'crs_ent':              1,
    #         'clst':                 1*12*-0.8,
    #         'sep':                  1*30*0.08,
    #         'l1':                   1e-3,
    #     }],
    #     'warm_optimizer_lrs': [{
    #         'prototype_vectors':    0.0007, #random.uniform(0.0001, 0.001) # 4e-2
    #     }], 
    #     'last_layer_lr':  [0.00065], #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
    #     'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
    #     'push_gap':          [10, 12, 14], # 17 #random.randint(10, 20)# 1_000_000 # not set
    #     'push_start':               [10, 15, 20, 25, 30], #25 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
    #     # BELOW IS UNUSED
    #     'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'joint_optimizer_lrs': [{ # learning rates for the different stages
    #         'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
    #         'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
    #     }]
    # }

    # # search 2/27/24
    # # These two are also hyperparameters. Feel free to add more values to try.
    # num_ptypes_per_class = [2] #random.randint(1, 3) # not set
    # ptype_length = [25] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    # hyperparameters = {    
    #     # comments after the line indicate jon's original settings
    #     # if the settings were not applicable, I write "not set".

    #     'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
    #     'latent_weight':            [0.9], #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
    #     'joint_weight_decay':       [0.065], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
    #     'gamma':                    [1], #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
    #     'warm_lr_step_size':        [1_000_000], # try 14? #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'crs_ent_weight':           [1],  # explore 3-4 powers of 2 in either direction
    #     'clst_weight':              [12*-0.8], # OG: 1*12*-0.8 times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
    #     'sep_weight':               [30*0.08], # OG: 1*30*0.08 go as high as 50x
    #     # 'clst_weight':              [1/8*12*-0.8, 1/4*12*-0.8, 1/2*12*-0.8, 12*-0.8, 2*12*-0.8, 4*12*-0.8, 8*12*-0.8], # OG: 1*12*-0.8 times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
    #     # 'sep_weight':               [1/8*30*0.08, 1/4*30*0.08, 1/2*30*0.08, 1*30*0.08, 2*30*0.08, 4*30*0.08, 8*30*0.08], # OG: 1*30*0.08 go as high as 50x
    #     'l1_weight':                [1e-3],
    #     'warm_ptype_lr':            [0.0007], #random.uniform(0.0001, 0.001) # 4e-2 
    #     'last_layer_lr':  [0.001], #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
    #     'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
    #     'push_gap':                 [11], # 17 #random.randint(10, 20)# 1_000_000 # not set
    #     'push_start':               [10], #25 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
    #     'num_pushes':               [0,1,2,3],
    #     # BELOW IS UNUSED
    #     'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'joint_optimizer_lrs': [{ # learning rates for the different stages
    #         'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
    #         'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
    #     }]
    # }

    # # search 3/2/24
    # # These two are also hyperparameters. Feel free to add more values to try.
    # num_ptypes_per_class = [2] #random.randint(1, 3) # not set
    # ptype_length = [25] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    # hyperparameters = {    
    #     # comments after the line indicate jon's original settings
    #     # if the settings were not applicable, I write "not set".

    #     'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
    #     'latent_weight':            [0.7, 0.9], #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
    #     'joint_weight_decay':       [0.0001, 0.001, 0.01, 0.1], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
    #     'gamma':                    [1], #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
    #     'warm_lr_step_size':        [1_000_000], # try 14? #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'crs_ent_weight':           [1],  # explore 3-4 powers of 2 in either direction
    #     # 'clst_weight':              [-4*12*-0.8], # OG: 1*12*-0.8 times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
    #     # 'sep_weight':               [-4*30*0.08], # OG: 1*30*0.08 go as high as 50x
    #     'clst_weight':              [1/32*12*-0.8, 1/16*12*-0.8, 1/8*12*-0.8, 1/4*12*-0.8, 1/2*12*-0.8, 12*-0.8, 2*12*-0.8, 4*12*-0.8, 8*12*-0.8], # OG: 1*12*-0.8 times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
    #     'sep_weight':               [1/8*30*0.08, 1/4*30*0.08, 1/2*30*0.08, 1*30*0.08, 2*30*0.08, 4*30*0.08, 8*30*0.08], # OG: 1*30*0.08 go as high as 50x
    #     'l1_weight':                [0.1, 0.01, 0.001, 0.0001], # 1e-3
    #     'warm_ptype_lr':            [0.005, 0.001, 0.0005, 0.0001], #random.uniform(0.0001, 0.001) # 4e-2 
    #     'last_layer_lr':  [0.005, 0.001, 0.0005, 0.0001], #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
    #     'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
    #     'push_gap':                 [11, 17], # 17 #random.randint(10, 20)# 1_000_000 # not set
    #     'push_start':               [10, 20, 30], #25 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
    #     'num_pushes':               [0,1,2,3], # [0,1,2,3], # number of push epochs, excluding final push
    #     # BELOW IS UNUSED
    #     'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set
    #     'joint_optimizer_lrs': [{ # learning rates for the different stages
    #         'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
    #         'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
    #     }]
    # }

    # # 40k model grid search (expected 600 models) 3/12/24
    # # These two are also hyperparameters. Feel free to add more values to try.
    # num_ptypes_per_class = [2,3] #random.randint(1, 3) # not set
    # ptype_length = [23,25,27] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    # hyperparameters = {
    #     # comments after the line indicate jon's original settings
    #     # if the settings were not applicable, I write "not set".

    #     'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
    #     'latent_weight':            [0.6, 0.7, 0.8, 0.9], #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
    #     'joint_weight_decay':       [0.065], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
    #     'gamma':                    [.1], #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
    #     'warm_lr_step_size':        [10], # try 14? #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'crs_ent_weight':           [1],  # explore 3-4 powers of 2 in either direction
    #     # 'clst_weight':              [12*-0.8], # OG: 1*12*-0.8 times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
    #     # 'sep_weight':               [30*0.08], # OG: 1*30*0.08 go as high as 50x
    #     'clst_weight':              [1/8*12*-0.8, 1/4*12*-0.8, 1/2*12*-0.8, 12*-0.8, 2*12*-0.8, 4*12*-0.8, 8*12*-0.8], # OG: 1*12*-0.8 times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
    #     'sep_weight':               [1/8*30*0.08, 1/4*30*0.08, 1/2*30*0.08, 1*30*0.08, 2*30*0.08, 4*30*0.08, 8*30*0.08], # OG: 1*30*0.08 go as high as 50x
    #     'l1_weight':                [1e-3],
    #     'warm_ptype_lr':            [0.08, 0.008, 0.0008], #random.uniform(0.0001, 0.001) # 4e-2 
    #     'last_layer_lr':  [0.001], #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
    #     'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
    #     'push_gap':                 [10, 15, 20], # 17 #random.randint(10, 20)# 1_000_000 # not set
    #     'push_start':               [35], #25 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
    #     'num_pushes':               [1, 2, 3, 4],
    #     # BELOW IS UNUSED
    #     'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'joint_optimizer_lrs': [{ # learning rates for the different stages
    #         'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
    #         'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
    #     }]
    # }

    # # manual search 3/13/24
    # # These two are also hyperparameters. Feel free to add more values to try.
    # num_ptypes_per_class = [3] #random.randint(1, 3) # not set, 3 was better than 2
    # ptype_length = [29] #[21, 25, 29] #[17, 19, 21, 25, 27, 29] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    # hyperparameters = {
    #     # comments after the line indicate jon's original settings
    #     # if the settings were not applicable, I write "not set".

    #     'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
    #     'latent_weight':            [0.9], #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
    #     'joint_weight_decay':       [0], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
    #     'gamma':                    [.8], #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
    #     'warm_lr_step_size':        [20*train.shape[0]//config['train_batch_size']], #random.randint(1, 20) # not set, how many BATCHES to cover before updating lr
    #     'crs_ent_weight':           [1],  # explore 3-4 powers of 2 in either direction
    #     'clst_weight':              [12*-0.8], # OG: 1*12*-0.8 times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
    #     'sep_weight':               [30*0.08], # OG: 1*30*0.08 go as high as 50x
    #     'l1_weight':                [1e-3],
    #     'warm_ptype_lr':            [0.1], #[0.5, 0.1, 0.05], # 0.7,0.07 #random.uniform(0.0001, 0.001) # 4e-2 
    #     'last_layer_lr':            [0.05], #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
    #     'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
    #     'push_gap':                 [100], # 17 #random.randint(10, 20)# 1_000_000 # not set
    #     'push_start':               [50], #25, 38 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
    #     'num_pushes':               [10], # 3-5?
    #     'last_layer_epochs':        [50], # 50
    #     # BELOW IS UNUSED
    #     'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'joint_optimizer_lrs': [{ # learning rates for the different stages
    #         'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
    #         'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
    #     }]
    # }

    # # 3/23/24 finding good warm lr, no pushes
    # # These two are also hyperparameters. Feel free to add more values to try.
    # num_ptypes_per_class = [3] #random.randint(1, 3) # not set, 3 was better than 2
    # ptype_length = [29] #[21, 25, 29] #[17, 19, 21, 25, 27, 29] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    # hyperparameters = {
    #     # comments after the line indicate jon's original settings
    #     # if the settings were not applicable, I write "not set".

    #     'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
    #     'latent_weight':            [0.9], #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
    #     'joint_weight_decay':       [-1], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
    #     'gamma':                    [.8, 0.5], #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
    #     'warm_lr_step_size':        [20*train.shape[0]//config['train_batch_size']], #random.randint(1, 20) # not set, how many BATCHES to cover before updating lr
    #     'crs_ent_weight':           [1],  # explore 3-4 powers of 2 in either direction
    #     'clst_weight':              [12*-0.8], # OG: 1*12*-0.8 times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
    #     'sep_weight':               [30*0.08], # OG: 1*30*0.08 go as high as 50x
    #     'l1_weight':                [1e-3],
    #     'warm_ptype_lr':            [0.1], #[0.5, 0.1, 0.05], # 0.7,0.07 #random.uniform(0.0001, 0.001) # 4e-2 
    #     'last_layer_lr':            [0.05], #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
    #     'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
    #     'push_gap':                 [-1], # 17 #random.randint(10, 20)# 1_000_000 # not set
    #     'push_start':               [200], #25, 38 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
    #     'num_pushes':               [0], # 3-5?
    #     'last_layer_epochs':        [0], # 50
    #     # BELOW IS UNUSED
    #     'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'joint_optimizer_lrs': [{ # learning rates for the different stages
    #         'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
    #         'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
    #     }]
    # }

    # 3/23/24 getting last layer lr and epochs
    # These two are also hyperparameters. Feel free to add more values to try.
    num_ptypes_per_class = [3] #random.randint(1, 3) # not set, 3 was better than 2
    ptype_length = [29] #[21, 25, 29] #[17, 19, 21, 23, 25, 27, 29] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    hyperparameters = {
        # comments after the line indicate jon's original settings
        # if the settings were not applicable, I write "not set".

        'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
        'latent_weight':            [0.8], #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
        'joint_weight_decay':       [-1], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
        'gamma':                    [-1], #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
        'warm_lr_step_size':        [1_000_000*train.shape[0]//config['train_batch_size']], #20 #random.randint(1, 20) # not set, how many BATCHES to cover before updating lr
        'crs_ent_weight':           [1],  # explore 3-4 powers of 2 in either direction
        'clst_weight':              [-1], #[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],#[10*12*-0.8, 1*12*-0.8, 0.1*12*-0.8], # OG: [12*-0.8], times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
        'sep_weight':               [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1], #[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],#[10*30*0.08, 1*30*0.08, 0.1*30*0.08], # OG: [30*0.08], go as high as 50x
        'l1_weight':                [0.001], #[10, 1, 0.1, 0.01, 0.001],
        'warm_ptype_lr':            [0.4], # first layer: 0.1 to 0.5 (0.4) #[0.5, 0.1, 0.05], # 0.7,0.07 #random.uniform(0.0001, 0.001) # 4e-2 
        'last_layer_lr':            [0.001], #[0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005], #[0.5, 0.01, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], # 0.001 was used, best? idk #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
        'last_layer_lr_after_second_push': [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
        'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
        'push_gap':                 [12], # 17 # random.randint(10, 20)# 1_000_000 # not set
        'push_start':               [12], # 13 for lr=0.1 #25, 38 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
        'num_pushes':               [2], # 3-5?
        'last_layer_epochs':        [85], # 50, 100
        # BELOW IS UNUSED
        'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
        'joint_optimizer_lrs': [{ # learning rates for the different stages
            'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
            'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
        }]
    }
    # end_epoch = params['push_start'] + params['push_gap'] * params['num_pushes']

    # 1. find a good warm lr and push_start by setting push_start to 300, last_layer_epochs to 0, and num_pushes to 0, and grid search through different warm_lrs.
    # 1.5. (optionally) modify gamma and warm_lr_step_size and gamma to improve even more, once you find good performance
    # 2. find a good last layer lr by setting last_layer_epochs to 300, with num_pushes at 0 (still using the warm lr and push_gap that you found)
    # 2.5. (optionally) modify the manual lr scheduler in the code to improve performance. possibly have different last layer epochs after the first one and after subsequent pushes?
    # 3. find a good push_gap by setting it to 300 and seeing how many epochs are necessary. num_pushes should be 1.
    # 4. find a good num_pushes by setting it to 10. If accuracy does not increase after each push, then consider lowering the learning rates by 75% after each push.
    
    # How far apart should cluster and separation be?
    # When and how should I find ideal cluster and separation weights?
    # - they are part of the loss fn for the warm ptype training and last layer training. 
    
    print(f"Hyperparameters: {hyperparameters}\n\n")

    # Generate all combinations of hyperparameters
    combinations = list(itertools.product(*hyperparameters.values()))

    # Iterate through all combinations
    combos = len(combinations)
    if combos > 1:
        print(f"\n\nExploring {combos} hyperparameter combinations for grid search.\n")
    for iter, combination in enumerate(combinations):
        params = dict(zip(hyperparameters.keys(), combination))
        print(f"\nAttempting combination {iter+1}/{combos}:")
        for key, value in params.items():
            print(f"{key}: {value}", flush=True)
        print('\n\n')
        params['coefs'] = {
            'crs_ent': params['crs_ent_weight'],
            'clst': params['clst_weight'],
            'sep': params['sep_weight'],
            'l1': params['l1_weight'],
        }
        # print(params['num_warm_epochs'])
        # pause = input("pause")

        # print('prototype_shape: {0}'.format(config['prototype_shape']))
        # print('Latent weight: {0}'.format(config['latent_weight']))

        # construct the model
        ppnet = ppn.construct_PPNet(
            features=backbone,
            pretrained=True,
            sequence_length=config['seq_target_length'],
            prototype_shape=params['prototype_shape'],
            num_classes=config['num_classes'],
            prototype_activation_function='this is not used',
            latent_weight=params['latent_weight'],
        )
        ppnet = ppnet.cuda()
        # ppnet_multi = torch.nn.DataParallel(ppnet) # uncommenting breaks it
        ppnet_multi = ppnet
        class_specific = True

        # define optimizer
        # weight decay is how much to penalize large weights in the network, 0-0.1
        joint_optimizer_specs = [
            {'params': ppnet.features.parameters(),
            'lr': params['joint_optimizer_lrs']['features'],
            'joint_weight_decay': params['joint_weight_decay']}, # bias are now also being regularized
            {'params': ppnet.prototype_vectors,
            'lr': params['joint_optimizer_lrs']['prototype_vectors']},
        ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        # after step-size epochs, the lr is multiplied by gamma
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=params['joint_lr_step_size'], gamma=params['gamma'])

        warm_optimizer_specs = [
            {'params': ppnet.prototype_vectors,
            'lr': params['warm_ptype_lr']}
        ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        # after step-size epochs, the lr is multiplied by gamma
        warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(warm_optimizer, step_size=params['warm_lr_step_size'], gamma=params['gamma'])
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(),
                                    'lr': params['last_layer_lr']}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
        # if start_push=17, push_gap=11, num_pushes=0,1,...,4
        # push 17(DONE)                                      17 + 0*11
        # push 17, push 22(DONE)                             17 + 1*11
        # push 17, push 22, push 33(DONE)                    17 + 2*11
        # push 17, push 22, push 33, push 44(DONE)           17 + 3*11
        # push 17, push 22, push 33, push 44, push 55(DONE)  17 + 4*11        

        """
        3/26/24
        Cluster and separation weights should be between -1 and 1
        - try shorter prototypes again
        - try different combos of clst and sep (ex. clst=-1, sep=0.1)
            - []
        - try a couple epochs of joint training 
            - joint training does not train the last layer
            - backbone should have a smaller lr, prototypes should have a larger
        """

        # cluster should be larger than separation. if it is lower, then it
        # will destroy accuracy on the push
          
        end_epoch = params['push_start'] + params['push_gap'] * params['num_pushes']
        print(f"End epoch: {end_epoch}")

        flush = True
        
        for epoch in range(30_000):
        # for epoch in tqdm(range(30_000)):
            if epoch == params['push_start'] + params['push_gap']*2:
                for param_group in last_layer_optimizer.param_groups:
                    param_group['lr'] = params['last_layer_lr_after_second_push']

            # Instead of manual lr below, use gamma in settings
            # if epoch == 0:
            #     for param_group in warm_optimizer.param_groups:
            #         param_group['lr'] *= 10
            # elif epoch == 10:
            #     for param_group in warm_optimizer.param_groups:
            #         param_group['lr'] /= 10
            # elif epoch == 20:
            #     for param_group in warm_optimizer.param_groups:
            #         param_group['lr'] /= 10

            # print(f"Calculating validation accuracy for epoch")
            val_actual, val_predicted, val_ptype_results  = tnt.test(
                model=ppnet_multi,
                dataloader=valloader,
                class_specific=class_specific,
                log=log
            )
            val_acc = metrics.accuracy_score(val_actual, val_predicted)
            # early_stopper(val_acc)
            print(f"Val acc before epoch {epoch}: {val_acc}", flush=flush)
            
            # peaked around epoch [38*, 58, 73]
            # if epoch >= 55: # 45, expected: 38
            #     # give up on testing the model
            #     print("giving up on achieving desired accuracy")
            #     break
            # if epoch >= 31 or val_acc >= 0.99: 
            if epoch == end_epoch:
                print(f"Stopping after epoch {epoch+1}.\n"
                    f"Final validation accuracy before push: {val_acc*100}%")
                print(f"Pushing prototypes since finished training")
                push_prototypes(
                    pushloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    preprocess_input_function=None, # normalize if needed
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here # sam: previously seq_dir
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    log=log
                )
                # evaluate on train, find aggregate sep and cluster loss for analysis purposes
                # print(f"Evaluating after push, before retraining last layer")
                train_actual, train_predicted, train_ptype_results = tnt.test(
                    model=ppnet_multi,
                    dataloader=trainloader,
                    class_specific=class_specific,
                    log=log
                )
                acc = np.mean(train_actual == train_predicted)
                print(f"After push, before retraining last layer:")
                print(f"\tTrain acc: {acc}")
                print(f"\tTrain Cluster: {train_ptype_results['cluster']}")
                print(f"\tTrainSeparation: {train_ptype_results['separation']}")
                # print(f"\tTrain prototype results, before retraining last layer: {train_ptype_results}")

                val_actual, val_predicted, val_ptype_results  = tnt.test(
                model=ppnet_multi,
                dataloader=valloader,
                class_specific=class_specific,
                log=log
                )
                val_acc = metrics.accuracy_score(val_actual, val_predicted)
                print(f"(Directly after push) Val acc at iteration 0: {val_acc}", flush=flush)

                # After pushing, retrain the last layer to produce good results again.
                tnt.last_only(model=ppnet_multi, log=log)
                print(f"Retraining last layer")
                # # Set the last layer lr to the original lr
                # for param_group in last_layer_optimizer.param_groups:
                #     param_group['lr'] = params['last_layer_lr']

                for param_group in last_layer_optimizer.param_groups:
                        print(f"Last layer lr: {param_group['lr']}")

                for i in range(params['last_layer_epochs']):
                    # if i == 25:
                    #     for param_group in last_layer_optimizer.param_groups:
                    #         param_group['lr'] *= 0.8 # was /= 5 for all three, 25, 35, 45
                    # elif i == 35:
                    #     for param_group in last_layer_optimizer.param_groups:
                    #         param_group['lr'] *= 0.8
                    # elif i == 45:
                    #     for param_group in last_layer_optimizer.param_groups:
                    #         param_group['lr'] *= 0.8

                    actual, pred, _ = tnt.train(
                        model=ppnet_multi,
                        dataloader=trainloader,
                        optimizer=last_layer_optimizer,
                        class_specific=class_specific,
                        coefs=params['coefs'],
                        log=log
                    )
                    acc = np.mean(actual == pred)
                    print(f"\tTrain acc at iteration {i+1}: {acc}", flush=flush)

                    val_actual, val_predicted, val_ptype_results  = tnt.test(
                        model=ppnet_multi,
                        dataloader=valloader,
                        class_specific=class_specific,
                        log=log
                    )
                    val_acc = metrics.accuracy_score(val_actual, val_predicted)
                    print(f"\tVal acc at iteration {i}: {val_acc}", flush=flush)

                # Get the final model validation and test scores
                print(f"Getting final validation and test accuracy after training.")
                val_actual, val_predicted, val_ptype_results  = tnt.test(
                    model=ppnet_multi,
                    dataloader=valloader,
                    class_specific=class_specific,
                    log=log
                )
                test_actual, test_predicted, test_ptype_results = tnt.test(
                    model=ppnet_multi,
                    dataloader=testloader,
                    class_specific=class_specific,
                    log=log
                )

                # Compute metrics for validation set
                print(f"Computing and storing results metrics")
                val_m_f1 = metrics.f1_score(val_actual, val_predicted, average='macro')
                val_m_recall = metrics.recall_score(val_actual, val_predicted, average='macro', zero_division=1)
                val_acc = metrics.accuracy_score(val_actual, val_predicted)
                val_m_precision = metrics.precision_score(val_actual, val_predicted, average='macro', zero_division=1)
                val_w_precision = metrics.precision_score(val_actual, val_predicted, average='weighted', zero_division=1)
                val_w_recall = metrics.recall_score(val_actual, val_predicted, average='weighted', zero_division=1)
                val_w_f1 = metrics.f1_score(val_actual, val_predicted, average='weighted')
                val_balanced_acc = metrics.balanced_accuracy_score(val_actual, val_predicted)

                # Compute metrics for test set
                test_m_f1 = metrics.f1_score(test_actual, test_predicted, average='macro')
                test_m_recall = metrics.recall_score(test_actual, test_predicted, average='macro', zero_division=1)
                test_acc = metrics.accuracy_score(test_actual, test_predicted)
                test_m_precision = metrics.precision_score(test_actual, test_predicted, average='macro', zero_division=1)
                test_w_precision = metrics.precision_score(test_actual, test_predicted, average='weighted', zero_division=1)
                test_w_recall = metrics.recall_score(test_actual, test_predicted, average='weighted', zero_division=1)
                test_w_f1 = metrics.f1_score(test_actual, test_predicted, average='weighted')
                test_bal_acc = metrics.balanced_accuracy_score(test_actual, test_predicted)

                print(f"Final val macro f1-score: {val_m_f1}")
                print(f"Final val micro accuracy: {val_acc}")
                print(f"Final test macro f1-score: {test_m_f1}")
                print(f"Final test micro accuracy: {test_acc}")

                print(f"Final validation:")
                print(f"\tCluster: {val_ptype_results['cluster']}")
                print(f"\tSeparation: {val_ptype_results['separation']}")
                # print(f"Final val prototype results: {val_ptype_results}")
                print(f"Final test:")
                print(f"\tCluster: {test_ptype_results['cluster']}")
                print(f"\tSeparation: {test_ptype_results['separation']}")
                # print(f"Final test prototype results: {test_ptype_results}")

                results = {
                    # Results
                    'val_macro_f1-score': val_m_f1,
                    'val_macro_recall': val_m_recall, 
                    'val_micro_accuracy': val_acc,
                    'val_macro_precision': val_m_precision,
                    'val_weighted_precision': val_w_precision, 
                    'val_weighted_recall': val_w_recall,
                    'val_weighted_f1-score': val_w_f1,
                    'val_balanced_accuracy': val_balanced_acc,
                    'test_macro_f1-score': test_m_f1,
                    'test_macro_recall': test_m_recall,
                    'test_micro_accuracy': test_acc,
                    'test_macro_precision': test_m_precision,
                    'test_weighted_precision': test_w_precision, 
                    'test_weighted_recall': test_w_recall,
                    'test_weighted_f1-score': test_w_f1,
                    'test_balanced_accuracy': test_bal_acc,
                    'epochs_taken': epoch+1,
                    'test_time': test_ptype_results['time'],
                    # Prototype Results
                    'val_cross_ent': val_ptype_results['cross_ent'],
                    'val_cluster': val_ptype_results['cluster'],
                    'val_separation': val_ptype_results['separation'],
                    'val_avg_separation': val_ptype_results['avg_separation'],
                    'val_p_avg_pair_dist': val_ptype_results['p_avg_pair_dist'],


                    # Variable variables
                    'num_ptypes_per_class': params['prototype_shape'][0] / config['num_classes'], # num_ptypes_per_class,
                    'ptype_length': params['prototype_shape'][2], # ptype_length,
                    'prototype_shape': params['prototype_shape'],
                    'ptype_activation_fn': 'unused',
                    'latent_weight': params['latent_weight'],
                    # joint is not used currently
                    'joint_features_lr': params['joint_optimizer_lrs']['features'],
                    'joint_ptypes_lr': params['joint_optimizer_lrs']['prototype_vectors'],
                    'warm_ptypes_lr': params['warm_ptype_lr'],
                    'last_layer_lr': params['last_layer_lr'],
                    'joint_weight_decay': params['joint_weight_decay'],
                    'joint_lr_step_size': params['joint_lr_step_size'],
                    'warm_lr_step_size': params['warm_lr_step_size'],
                    'cross_entropy_weight': params['coefs']['crs_ent'],
                    'cluster_weight': params['coefs']['clst'],
                    'separation_weight': params['coefs']['sep'],
                    'l1_weight': params['coefs']['l1'],
                    'num_warm_epochs': params['num_warm_epochs'],
                    'push_gap': params['push_gap'],
                    'push_start': params['push_start'],
                    'num_pushes': params['num_pushes'],
                    'last_layer_epochs': params['last_layer_epochs'],

                    # Static Variables
                    'seq_count_thresh': config['seq_count_thresh'],           
                    'trainRandomInsertions': config['trainRandomInsertions'],  
                    'trainRandomDeletions': config['trainRandomDeletions'],  
                    'trainMutationRate': config['trainMutationRate'],      
                    'oversample': config['oversample'],               
                    'encoding_mode': config['encoding_mode'],    
                    'push_encoding_mode': config['push_encoding_mode'],  
                    'applying_on_raw_data': config['applying_on_raw_data'],
                    'augment_test_data': config['augment_test_data'],
                    'load_existing_train_test': config['load_existing_train_test'], 
                    'train_batch_size': config['train_batch_size'], 
                    'test_batch_size': config['test_batch_size'],
                    'num_classes': config['num_classes'],
                    'seq_target_length': config['seq_target_length'],   
                    'addTagAndPrimer': config['addTagAndPrimer'],
                    'addRevComplements': config['addRevComplements'],
                    'val_portion_of_train': val_portion,
                    # 'patience': early_stopper.patience,
                    # 'min_pc_improvement': early_stopper.min_pct_improvement,
                }
                utils.update_results(
                    results,
                    compare_cols='ppn',
                    model=ppnet_multi,
                    filename='search_for_warm_improvement_3_18_24.csv',
                    save_model_dir = None
                    # save_model_dir='saved_ppn_models'
                )
                break # for early stopping

            elif epoch >= params['push_start'] and (epoch - params['push_start']) % params['push_gap'] == 0:
                print(f"Push epoch")
                push_prototypes(
                    pushloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    preprocess_input_function=None, # normalize if needed
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here # sam: previously seq_dir
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    log=log,
                    sanity_check=False
                )

                # evaluate on train, find aggregate sep and cluster loss for analysis purposes
                # print(f"Evaluating after push, before retraining last layer")
                train_actual, train_predicted, train_ptype_results = tnt.test(
                    model=ppnet_multi,
                    dataloader=trainloader,
                    class_specific=class_specific,
                    log=log
                )
                acc = np.mean(train_actual == train_predicted)
                print(f"After push, before retraining last layer:")
                print(f"\tTrain acc: {acc}")
                print(f"\tTrain Cluster: {train_ptype_results['cluster']}")
                print(f"\tTrain Separation: {train_ptype_results['separation']}")
                # print(f"\tTrain prototype results, before retraining last layer: {train_ptype_results}")

                val_actual, val_predicted, val_ptype_results  = tnt.test(
                model=ppnet_multi,
                dataloader=valloader,
                class_specific=class_specific,
                log=log
                )
                val_acc = metrics.accuracy_score(val_actual, val_predicted)
                print(f"(Directly after push) Val acc at iteration 0: {val_acc}", flush=flush)

                # After pushing, retrain the last layer to produce good results again.
                tnt.last_only(model=ppnet_multi, log=log)
                print(f"Retraining last layer: ")
                # # Set the last layer lr to the original lr
                # for param_group in last_layer_optimizer.param_groups:
                #     param_group['lr'] = params['last_layer_lr']

                for param_group in last_layer_optimizer.param_groups:
                        print(f"Last layer lr: {param_group['lr']}")

                for i in range(params['last_layer_epochs']):
                    # if i == 25:
                    #     for param_group in last_layer_optimizer.param_groups:
                    #         param_group['lr'] /= 5
                    # elif i == 35:
                    #     for param_group in last_layer_optimizer.param_groups:
                    #         param_group['lr'] /= 5
                    # elif i == 45:
                    #     for param_group in last_layer_optimizer.param_groups:
                    #         param_group['lr'] /= 5

                    actual, pred, _ = tnt.train(
                        model=ppnet_multi,
                        dataloader=trainloader,
                        optimizer=last_layer_optimizer,
                        class_specific=class_specific,
                        coefs=params['coefs'],
                        log=log
                    )
                    # acc = np.mean(actual == pred)
                    # print(f"\tTrain acc at iteration {i}: {acc}", flush=flush)

                    val_actual, val_predicted, val_ptype_results  = tnt.test(
                        model=ppnet_multi,
                        dataloader=valloader,
                        class_specific=class_specific,
                        log=log
                    )
                    val_acc = metrics.accuracy_score(val_actual, val_predicted)
                    print(f"\tVal acc at iteration {i}: {val_acc}", flush=flush)
                    

                # # Reset warm lr to original lr
                # for param_group in warm_optimizer.param_groups:
                #     param_group['lr'] = params['warm_ptype_lr']

                # Lower the warm and last_layer lr by half after each push
                for param_group in warm_optimizer.param_groups:
                    param_group['lr'] *= 0.7
                for param_group in last_layer_optimizer.param_groups:
                    param_group['lr'] *= 0.7

            elif epoch < params['num_warm_epochs']:
                # print(f"Train epoch")
                # train the prototypes without modifying the backbone
                tnt.warm_only(model=ppnet_multi, log=log)
                _, _, ptype_results = tnt.train(
                    model=ppnet_multi,
                    dataloader=trainloader,
                    optimizer=warm_optimizer,
                    scheduler=warm_lr_scheduler,
                    class_specific=class_specific,
                    coefs=params['coefs'],
                    log=log
                )

                for param_group in warm_optimizer.param_groups:
                    print(f"Warm optimizer lr: {param_group['lr']}")
                # print(f"Prototype results: {ptype_results}")
                    
            else:
                # train the prototypes and the backbone
                tnt.joint(model=ppnet_multi, log=log)
                _, _, ptype_results = tnt.train(
                    model=ppnet_multi,
                    dataloader=trainloader,
                    optimizer=joint_optimizer,
                    scheduler=joint_lr_scheduler,
                    class_specific=class_specific,
                    coefs=params['coefs'],
                    log=log
                )   
                # print(f"Prototype results: {ptype_results}")
            

        # end of training and testing for given model
        del ppnet
        del ppnet_multi
        del warm_optimizer
        del warm_optimizer_specs
        del joint_optimizer
        del joint_optimizer_specs
        del joint_lr_scheduler
        del warm_lr_scheduler
        del last_layer_optimizer
        del last_layer_optimizer_specs
    # end of grid search
# end of number of trials
print(f"Finished search.")
