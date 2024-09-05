import argparse
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
import glob
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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# adding the parent folder to the system path so we can import from it
sys.path.append('..')
import utils
import models
import ppnet as ppn
from dataset import Sequence_Data
from torch.utils.data import DataLoader

def conditionally_save_model(model, model_dir, arr_job_id, comb_num, accu, target_accu, log=print):
    if accu < target_accu:
        # Do not save
        return

    # Define the path for the new model
    new_model_path = os.path.join(model_dir, (str(arr_job_id) + '_' + str(comb_num) + '_{0:.4f}.pth').format(accu))

    # Get all the existing models
    existing_models = glob.glob(os.path.join(model_dir, str(arr_job_id) + '_' + str(comb_num) + '_*.pth'))

    # If there are no existing models, save the new model
    if not existing_models:
        torch.save(obj=model, f=new_model_path)
        log('\tSaved, since above {0:.2f}%'.format(target_accu * 100))
        return

    # Get the accuracy of the existing model with the highest accuracy
    existing_accu = max(float(model_path.split('_')[-1].replace('.pth', '')) for model_path in existing_models)

    # If the accuracy of the new model is higher than the existing model with the highest accuracy
    if accu > existing_accu:
        # Delete the existing model with the highest accuracy
        os.remove(os.path.join(model_dir, (str(arr_job_id) + '_' + str(comb_num) + '_{0:.4f}.pth').format(existing_accu)))

        # Save the new model
        torch.save(obj=model, f=new_model_path)
        log('\tSaved, since accuracy {0:.2f}% is higher than existing accuracy {1:.2f}%'.format(accu * 100, existing_accu * 100))

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
    'train_path': '../datasets/train_same_as_zurich_oversampled_t70_noise-0_thresh-2.csv',
    'test_path': '../datasets/test_same_as_zurich.csv', # may optionally be overridden later
    # 'train_path': "../datasets/train_oversampled_t70_noise-0_thresh-2.csv", #'../datasets/train_dup.csv',
        # for training the protopnet, noise is added online *during* training,
        # not before in the csv. so, always use noise-0. Also, use train_dup.csv
        # instead of train_oversampled_t70_noise-0_thresh-2.csv because this
        # allows flexibility in length (insertions, deletions) before truncating.
    # 'test_path': '../datasets/test_t70_noise-2_thresh-2.csv', #'../datasets/test.csv', # OVERRIDDEN BY CMD LINE ARGUMENTS
    'species_col': 'species_cat',     # name of column containing species
    'seq_col': 'seq',                 # name of column containing sequences

    # Logged information (plus below):
    'verbose': True,
    'seq_count_thresh': 2,            # ex. keep species with >1 sequences
    'oversample': True,               # whether or not to oversample train # POSSIBLY OVERRIDDEN IN ARCH SEARCH
    'encoding_mode': 'probability',   # 'probability' or 'random'
    'push_encoding_mode': 'probability',   # 'probability' or 'random'
    # Whether or not applying on raw unlabeled data or "clean" ref db data.
    'applying_on_raw_data': False,
    # Whether or not to augment the test set.
    # 'load_existing_train_test': True, # use the same train/test split as Zurich, already saved in two different csv files/ COMMENTED BECAUSE TRUE BY DEFAULT
    'train_batch_size': 156, # 780 oversampled. prev. 64. 1,2,3,4,5,6,10,12,13,15,20,26,30,39,52,60,65,78,130,156,195,260,390,=780
    'test_batch_size': 35, # 175. 1, 2, 4, 47, 94, 188 NOT # 1,5,7,25,35,175
    'push_batch_size': 187, # 748 different from train since train is oversampled.
    'num_classes': 156
}

if config['applying_on_raw_data']:
    config['seq_target_length'] = 150
    config['addTagAndPrimer'] = True 
    config['addRevComplements'] = True
elif not config['applying_on_raw_data']:
    config['seq_target_length'] = 70        # 70 (prev. 71) or 60 or 64
    config['addTagAndPrimer'] = False
    config['addRevComplements'] = False
assert config['seq_target_length'] % 2 == 0, \
    "Error: sequence length must be even"

# As of 8/3/24, when running on different levels of noise, we set the train_path
# to be a raw (oversampled) dataset with NO noise, then add noise in the online
# augmentation. We set the test path to be a pre-created test set with noise
# and have NO online augmentation. We do this to allow simple reproducability
# on the test set without having to use random seeds to reproduce the data.


# allow the user to call the program with command line gpu argument
# ex. python3 main.py -gpuid=0,1,2,3
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
parser.add_argument('--job_id', nargs=1, type=int, default=-1) # not used as of 4/5/23
parser.add_argument('--arr_job_id', nargs=1, type=int, default=-1) # same for all elements in the array. used as the name if saving the model 
parser.add_argument('--comb_num', nargs=1, type=int, default=-1)
parser.add_argument('--train_noise', nargs=1, type=int, default=1)
parser.add_argument('--test_noise', nargs=1, type=int, default=1)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(f"Available CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
try:
    args.comb_num = args.comb_num[0]
except:
    pass
try:
    args.arr_job_id = args.arr_job_id[0]
except:
    pass
try:
    args.job_id = args.job_id[0]
except:
    pass
try:
    print(f"comb_num: {args.comb_num}")
    print(f"arr_job_id: {args.arr_job_id}")
    print(f"job_id: {args.job_id}")
except:
    pass
try:
    train_noise = args.train_noise[0]
    test_noise = args.test_noise[0]
except:
    pass

if train_noise == 0:
    config['trainRandomInsertions'] = [0, 0]
    config['trainRandomDeletions'] =  [0, 0]
    config['trainMutationRate'] =  0
elif train_noise == 1:
    config['trainRandomInsertions'] =  [0, 2]
    config['trainRandomDeletions'] =  [0, 2]
    config['trainMutationRate'] =  0.05
elif train_noise == 2:
    config['trainRandomInsertions'] =  [0, 4]
    config['trainRandomDeletions'] =  [0, 4]
    config['trainMutationRate'] =  0.1

# Either 
# 1. add online augmentation to the test set by using the three 'if' stmts below, OR
# 2. use a CSV with noise already added by uncommenting the four lines below
config['test_path'] = f'../datasets/test_same_as_zurich_t70_noise-{test_noise}_thresh-2.csv'
config['testRandomInsertions'] = [0, 0]
config['testRandomDeletions'] =  [0, 0]
config['testMutationRate'] =  0

# if test_noise == 0:
#     config['testRandomInsertions'] = [0, 0]
#     config['testRandomDeletions'] =  [0, 0]
#     config['testMutationRate'] =  0
# elif test_noise == 1:
#     config['testRandomInsertions'] =  [1, 1]
#     config['testRandomDeletions'] =  [1, 1]
#     config['testMutationRate'] =  0.02
# elif test_noise == 2:
#     config['testRandomInsertions'] =  [2, 2]
#     config['testRandomDeletions'] =  [2, 2]
#     config['testMutationRate'] =  0.04

print(f"\nTraining on train_noise {train_noise} and test_noise {test_noise}")
print(f"Training on {config['train_path']}")
print(f"Testing on {config['test_path']}")
print(f"testRandomInsertions: {config['testRandomInsertions']}")
print(f"testRandomDeletions: {config['testRandomDeletions']}")
print(f"testMutationRate: {config['testMutationRate']}")
print(f"trainRandomInsertions: {config['trainRandomInsertions']}")
print(f"trainRandomDeletions: {config['trainRandomDeletions']}")
print(f"trainMutationRate: {config['trainMutationRate']}")





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

# <name redacted for submission purposes> says: we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)

# Unused since the number of pushes and the push start and gap defines how many
# epochs it runs.
# early_stopper = utils.EarlyStopping(
#     patience=7,
#     min_pct_improvement=1,#1,#3, # previously 20 epochs, 0.1% (for backbone)
#     verbose=False
# )

log = print
# random.seed(36)
warnings.filterwarnings('ignore', 'y_pred contains classes not in y_true')

# split data into train, validation, and test # as of July 2024, do not use a validation set # as of 8/26, use a val set
train = pd.read_csv(config['train_path'], sep=',')
test = pd.read_csv(config['test_path'], sep=',')
train_full = train.copy()

# Make sure that all classes in the test set are present in the train set
class_difference = set(test['species_cat'].unique()) - set(train['species_cat'].unique())
if not class_difference:
    print("All classes in the test set are present in the training set.")
else:
    print(f"WARNING: Classes are present in the test set that are not part of training set: {class_difference}")

# le = LabelEncoder()
# train[config['species_col']] = le.fit_transform(train[config['species_col']])
# test[config['species_col']] = le.transform(test[config['species_col']])
# dump(le, './datasets/label_encoder.joblib')

val_portion = 0.2
# If you were to use a validation set, use Fluck's stratified split method, rather than sklearn's method
train, val = utils.stratified_split(
    train,
    config['species_col'],
    val_portion
)
# Before 8/26/24, it was:
# train, test = utils.stratified_split(
#     df,
#     config['species_col'],
#     config['test_split']
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     train[config['seq_col']], # X
#     train[config['species_col']], # y
#     test_size=val_portion,
#     random_state=42,
#     stratify=train[config['species_col']]
# )
# print(X_train.shape)
# print(y_train.shape)
# print(type(X_train))
# print(type(y_train))

# COMMENTED the line below because I only want to use a train/test split not a
# train/val/test split. However, I don't want to delete validation
# functionality. For this reason, to just to a train/test, comment out the
# following line, and then ignore validation accuracy numbers. ALSO make the
# push dataset use the correct datasets, below.

# train = X_train.to_frame().join(y_train.to_frame())

print("Before oversampling:")
utils.print_descriptive_stats(train, ['species_cat'])
if config['oversample']:
    train = utils.oversample_underrepresented_species(
        train,
        config['species_col'],
        config['verbose']
    )
# print("After oversampling")
# utils.print_descriptive_stats(train, ['species_cat'])
# wait=input("PAUSE")

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
    val[config['seq_col']],
    val[config['species_col']],
    # X_val,
    # y_val,
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
    # USE THESE TWO lines if you're just doing a train/test split
    # train[config['seq_col']],
    # train[config['species_col']],
    # USE THESE TWO lines if you're doing a train/val/test split
    train_full[config['seq_col']],
    train_full[config['species_col']],
    # X_train,
    # y_train,
    insertions=[0,0],
    deletions=[0,0],
    mutation_rate=0,
    encoding_mode=config['push_encoding_mode'],
    seq_len=config['seq_target_length']
)
pushloader = DataLoader(
    push_dataset,
    batch_size=config['push_batch_size'],
    shuffle=False
)


def calculate_metrics(model, dataloader):
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to('cuda'), data[1].to('cuda')
            outputs = model(inputs) # returns (logits, max_similarities)
            _, predicted = torch.max(outputs[0], 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = (np.array(all_labels) == np.array(all_predictions)).mean()
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    return accuracy, precision, recall, f1

test_accuracy = -1 # to initialize and make it available after training


# UNCOMMENT BELOW TO EVALUATE A PRETRAINED SAVED MODEL (OR TO RUN PUSH FOR ANALYSIS)

# # model = torch.load('saved_ppn_models/1857326_0.9681.pth') # 93, 88 val, test acc
# # model = torch.load('saved_ppn_models/1857326_0.9787.pth') # 94, 88 val, test acc
# # model = torch.load('saved_ppn_models/1857326_0.9894.pth') # 95.2, 90.8 val, test acc
# # model = torch.load('saved_ppn_models/1857478_1.0000.pth') # (0.9627, 0.9468, 0.9574, 0.9627, 0.9574, avg = 0.9574) (0.9314, 0.9085, 0.92, 0.9314, 0.9371, avg=0.9257) val, test acc
# model = torch.load('saved_ppn_models/1878231_3_-1.pth') # 
# model.to('cuda')

# # Calculate and print the train, validation, and test metrics
# print("Calculating test metrics")
# test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(model, testloader)

# print(f'\nTest Accuracy: {test_accuracy}\n Test Precision: {test_precision}\n Test Recall: {test_recall}\n Test F1: {test_f1}')

# wait = input("PAUSE")

# # push_prototypes(
# #     pushloader, # pytorch dataloader (must be unnormalized in [0,1])
# #     prototype_network_parallel=model, # pytorch network with prototype_vectors
# #     preprocess_input_function=None, # normalize if needed
# #     root_dir_for_saving_prototypes='./saved_prototypes', # if not None, prototypes will be saved here # Note: previously seq_dir
# #     epoch_number=9999, # if not provided, prototypes saved previously will be overwritten
# #     log=log
# # )

# # wait = input("Pause")


# COMMENTED code that varies what pretrained backbone the ppnet uses

# if args.comb_num == 1:
#     model_path = "../backbone_1-layer_95.4.pt"
#     backbone = models.Small_Best_Updated()
# elif args.comb_num == 2:
#     model_path = "../backbone_2-layer_94.3.pt"
#     backbone = models.Small_Best_Updated_2layer()
# elif args.comb_num == 3:
#     model_path = "../backbone_3-layer_94.3.pt"
#     backbone = models.Small_Best_Updated_3layer()
# elif args.comb_num == 4:
#     model_path = "../backbone_4-layer_93.7.pt"
#     backbone = models.Small_Best_Updated_4layer()

# elif args.comb_num == -1:
#     model_path = "../backbone_3-layer_94.3.pt"
#     backbone = models.Small_Best_Updated_3layer()

# # 1,2,3,4,5 -> 1 layer.   6,7,8,9,10 -> 2 layer backbone.
# if args.comb_num < 6:
#     model_path = "../backbone_1-layer_95.4.pt"
#     backbone = models.Small_Best_Updated()
# elif args.comb_num >= 6:
#     model_path = "../backbone_2-layer_94.3.pt"
#     backbone = models.Small_Best_Updated_2layer()

model_path = "../backbone_1-layer_95.4.pt"
backbone = models.Small_Best_Updated()


# model_path = "../best_model_20240103_210217.pt" # updated large best
# model_path = "../best_model_20240111_060457.pt" # small best with pool=3, stride=1
# model_path = "../best_model_20240126_120130.pt" # small best with pool=2,stride=2
# model_path = "../small_best_updated_backbone.pt" # small best with pool=2,stride=2
# large_best = models.Large_Best()

backbone.load_state_dict(torch.load(model_path))
# Remember to comment out the linear layer in the forward function

# begin hyperparameter search
# this is the number of times you want to repeat either the
# grid search below, or the random search below.
#######################
num_trials = 2
#######################
val_accs = []
test_accs = []
for trial in range(num_trials):

    print(f"\n\nTrial {trial+1}\n")
    # early_stopper.reset()

    print('training set size: {0}'.format(len(trainloader.dataset)))
    print('push set size: {0}'.format(len(pushloader.dataset)))
    print('test set size: {0}'.format(len(testloader.dataset)))
    print('train batch size: {0}'.format(config['train_batch_size']))
    print('test batch size: {0}'.format(config['test_batch_size'])) # TODO: why is there a remainder of 13 if you print out x.shape in ppnet's forward function?

    num_latent_channels = 512
  
    """
    If you want to do GRID SEARCH, then set the number of trials above to 1
    and defined the lists of numbers to try below.

    If you want to do do RANDOM SEARCH, then set the number of trials above to
    however many combinations you want to try, and define the random
    distributions below.
    """

    # These two are also hyperparameters. Feel free to add more values to try.
    # end_epoch = params['push_start'] + params['push_gap'] * params['num_pushes']-1
    num_ptypes_per_class = [3] #random.randint(1, 3)
    ptype_length = [5] # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35] # BEST 8/4/24: [5] #[15] #[21, 25, 29] #[15, 17, 19, 21, 23, 25, 27, 29] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    hyperparameters = {
        # comments after the line indicate <name redacted for submission purposes>'s original settings
        # if the settings were not applicable, I write "not set".

        'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
        'latent_weight':            [0.7],                          # [0, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1],  #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
        

        'num_warm_epochs':          [35+4*45],                        # [0, 35+1*45, 35+2*45, 35+3*45, 35+4*45, 35+5*45],                    # random.randint(0, 10) # not set
        'push_start':               [15],  #35                          # 37 35 for 0.01, 0.8,20. 13 for lr=0.1 #25, 38 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
        'push_gap':                 [15, 30],  #45                         # 35, 17 # random.randint(10, 20)# 1_000_000 # not set
        'num_pushes':               [2,3,4], #4                            # 3-5?

        'crs_ent_weight':           [1],                            # explore 3-4 powers of 2 in either direction
        'clst_weight':              [-1],                      #[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],#[10*12*-0.8, 1*12*-0.8, 0.1*12*-0.8], # OG: [12*-0.8], times 0.13, 0.25, 0.5, 1, 2, 4, 8, 16, 32 times this value, # 50 *-0.8 and 100 * 0.08
        'sep_weight':               [0.3],                      #[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],#[10*30*0.08, 1*30*0.08, 0.1*30*0.08], # OG: [30*0.08], go as high as 50x
        'l1_weight':                [0.0001],  #0                     #[0, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005], [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1], #[10, 1, 0.1, 0.01, 0.001],
        
        'p0_warm_ptype_lr':         [0.05],                          #GOOD [0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5],                               # [0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5],                         # the warm prototype lr for before the first push 0.35 0.1 to 0.5 (0.4) #[0.5, 0.1, 0.05], # 0.7,0.07 #random.uniform(0.0001, 0.001) # 4e-2 
        'p0_warm_ptype_gamma':      [0.75],                          #[0.7, 0.75, 0.8, 0.83, 0.86, 0.9, 0.92, 0.94, 0.96, 0.98, 1],                                #[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1],                           #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
        'p0_warm_ptype_step_size':  [25],                            # train_shape[0] is 780, train_batch_size is 156 #20 #random.randint(1, 20) # not set, how many BATCHES to cover before updating lr
        # push 1
        'p1_last_layer_lr':         [0.001],                        #GOOD [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005], #[0.5, 0.01, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], # 0.001 was used, best? idk #random.uniform(0.0001, 0.001) # <name redacted for submission purposes>: 0.02, OG: 0.002
        'p1_last_layer_iterations': [15],   #80                        #GOOD 80-85 (based off jobs. changed based off visuals) for 0.001, 215 for 0.00075
        'p1_warm_ptype_lr':         [0.05],                         # the warm prototype lr for after the first push
        'p1_warm_ptype_gamma':      [0.9],
        'p1_warm_ptype_step_size':  [10],
        # push 2
        'p2_last_layer_lr':         [0.008],                            #[0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'p2_last_layer_iterations': [20],    # 80                       #
        'p2_warm_ptype_lr':         [0.05],                          # the warm prototype lr for after the second push
        'p2_warm_ptype_gamma':      [0.9],                          # 0.9, every 10
        'p2_warm_ptype_step_size':  [10],
        # push 3
        'p3_last_layer_lr':         [0.01],
        'p3_last_layer_iterations': [20],     #80                       # 
        'p3_warm_ptype_lr':         [0.025],                         # the warm prototype lr for after the third push
        'p3_warm_ptype_gamma':      [0.9],
        'p3_warm_ptype_step_size':  [10],
        # push 4
        'p4_last_layer_lr':         [0.0005],
        'p4_last_layer_iterations': [20],     #80                      # 
        'p4_warm_ptype_lr':         [0.02],                         # the warm prototype lr for after the third push
        'p4_warm_ptype_gamma':      [0.9],
        'p4_warm_ptype_step_size':  [10],
        # push 5
        'p5_last_layer_lr':         [0.0001],
        'p5_last_layer_iterations': [20],   #80

        'joint_weight_decay':       [0.000005],                           #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
        'joint_lr_step_size':       [10],                           #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
        'joint_gamma':              [0.7],
        'joint_feature_lr':         [0.0001],                           #[0.1, 0.01, 0.001, 0.0001, 0.00001], # should be lower than ptype lr 0.003
        'joint_ptype_lr':           [0.02]                             #[0.1, 0.01, 0.001, 0.0001, 0.00001]  # 0.003
    }

    hyperparameters['p0_warm_ptype_step_size'] = [x*train.shape[0]//config['train_batch_size'] for x in hyperparameters['p0_warm_ptype_step_size']]
    hyperparameters['p1_warm_ptype_step_size'] = [x*train.shape[0]//config['train_batch_size'] for x in hyperparameters['p1_warm_ptype_step_size']]
    hyperparameters['p2_warm_ptype_step_size'] = [x*train.shape[0]//config['train_batch_size'] for x in hyperparameters['p2_warm_ptype_step_size']]
    hyperparameters['p3_warm_ptype_step_size'] = [x*train.shape[0]//config['train_batch_size'] for x in hyperparameters['p3_warm_ptype_step_size']]
    hyperparameters['p4_warm_ptype_step_size'] = [x*train.shape[0]//config['train_batch_size'] for x in hyperparameters['p4_warm_ptype_step_size']]

    # 1. find a good warm lr and push_start by setting push_start to 300, last_layer_iterations to 0, and num_pushes to 0, and grid search through different warm_lrs.
    # 1.5. (optionally) modify gamma and warm_lr_step_size and gamma to improve even more, once you find good performance
    # 2. find a good last layer lr by setting last_layer_iterations to 300, with num_pushes at 0 (still using the warm lr and push_gap that you found)
    # 2.5. (optionally) modify the manual lr scheduler in the code to improve performance. possibly have different last layer epochs after the first one and after subsequent pushes?
    # 3. find a good push_gap by setting it to 300 and seeing how many epochs are necessary. num_pushes should be 1.
    # 4. find a good num_pushes by setting it to 10. If accuracy does not increase after each push, then consider lowering the learning rates by 75% after each push.
    
    # How far apart should cluster and separation be?
    # When and how should I find ideal cluster and separation weights?
    # - they are part of the loss fn for the warm ptype training and last layer training. 
    
    print(f"Hyperparameters: {hyperparameters}\n\n")
    ch = {key: value for key, value in hyperparameters.items() if len(value) > 1}
    print(f"Checking hyperparameters: {ch}\n\n")
    del ch
    if args.comb_num != -1:
        print(f"Performing combination number {args.comb_num} trial {trial}")
    # Generate all combinations of hyperparameters
    combinations = list(itertools.product(*hyperparameters.values()))
    combos = len(combinations)

    # Iterate through all combinations. 
    # If a particular array index is supplied, then only run that combination.
    print(f"\n\nExploring {combos} hyperparameter combination(s) for grid search.\n")
    for iter, combination in enumerate(combinations, start=1):
        # comb_num=-1 indicates that there was no combination_number supplied,
        # which means that it isn't being run in parallel using an array.
        # When evaluating different noise levels but not different hyperparameters,
        # set comb_num to 1 (or I believe -1 would work, too).
        if args.comb_num != -1: 
            if args.comb_num != iter:
                continue
        params = dict(zip(hyperparameters.keys(), combination))
        print(f"\nAttempting combination {iter}/{combos}:")
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
            'lr': params['joint_feature_lr'],
            'joint_weight_decay': params['joint_weight_decay']}, # bias are now also being regularized
            {'params': ppnet.prototype_vectors,
            'lr': params['joint_ptype_lr']},
        ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        # after step-size epochs, the lr is multiplied by gamma
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=params['joint_lr_step_size'], gamma=params['joint_gamma'])

        warm_optimizer_specs = [
            {'params': ppnet.prototype_vectors,
            'lr': params['p0_warm_ptype_lr']}
        ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        # after step-size epochs, the lr is multiplied by gamma
        warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(warm_optimizer, step_size=params['p0_warm_ptype_step_size'], gamma=params['p0_warm_ptype_gamma'])
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(),
                                    'lr': params['p1_last_layer_lr']}]
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
            - backbone should have a smaller lr, prototypes should have a larger lr
        - cluster should be larger than separation. if it is lower, then it
          will destroy accuracy on the push
        """
          
        end_epoch = params['push_start'] + params['push_gap'] * params['num_pushes']-1
        print(f"End epoch: {end_epoch}")

        flush = True

        pushes_completed = 0

        for epoch in range(30_000):
        # for epoch in tqdm(range(30_000)):

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

            # Update the number of
            # last layer iterations
            # last layer lr
            # based on the push num.
            if pushes_completed == 0:
                for param_group in warm_optimizer.param_groups:
                    param_group['lr'] = params['p0_warm_ptype_lr']
                last_layer_iterations = params['p1_last_layer_iterations']
                for param_group in last_layer_optimizer.param_groups:
                    param_group['lr'] = params['p1_last_layer_lr']
            elif pushes_completed == 1:
                for param_group in warm_optimizer.param_groups:
                    param_group['lr'] = params['p1_warm_ptype_lr']
                last_layer_iterations = params['p2_last_layer_iterations']
                for param_group in last_layer_optimizer.param_groups:
                    param_group['lr'] = params['p2_last_layer_lr']
            elif pushes_completed == 2:
                for param_group in warm_optimizer.param_groups:
                    param_group['lr'] = params['p2_warm_ptype_lr']
                last_layer_iterations = params['p3_last_layer_iterations']
                for param_group in last_layer_optimizer.param_groups:
                    param_group['lr'] = params['p3_last_layer_lr']
            elif pushes_completed == 3:
                for param_group in warm_optimizer.param_groups:
                    param_group['lr'] = params['p3_warm_ptype_lr']
                last_layer_iterations = params['p4_last_layer_iterations']
                for param_group in last_layer_optimizer.param_groups:
                    param_group['lr'] = params['p4_last_layer_lr']
            elif pushes_completed == 4:
                last_layer_iterations = params['p5_last_layer_iterations']
                for param_group in last_layer_optimizer.param_groups:
                    param_group['lr'] = params['p5_last_layer_lr']

            # if epoch >= 31 or val_acc >= 0.99: 
            if epoch == end_epoch:
                print(f"Stopping after epoch {epoch+1}.\n"
                    f"Final validation accuracy before push {pushes_completed + 1}: {val_acc}")
                push_prototypes(
                    pushloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    preprocess_input_function=None, # normalize if needed
                    root_dir_for_saving_prototypes= None, # os.path.join('saved_prototypes', f"{str(args.arr_job_id)}_{str(args.comb_num)}_-1_latent_{params['latent_weight']}"), # if not None, prototypes will be saved here # Note: previously seq_dir
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    log=log
                )
                pushes_completed += 1
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
                print(f"(Directly after push {pushes_completed}) Val acc at iteration 0: {val_acc}", flush=flush)

                # After pushing, retrain the last layer to produce good results again.
                tnt.last_only(model=ppnet_multi, log=log)
                print(f"Retraining last layer")

                # Print out the last layer lr
                for param_group in last_layer_optimizer.param_groups:
                        print(f"Last layer lr: {param_group['lr']}")

                for i in range(last_layer_iterations):

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

                    # conditionally_save_model(
                    #     ppnet,
                    #     'saved_ppn_models',
                    #     arr_job_id=str(args.arr_job_id),
                    #     comb_num=str(args.comb_num),
                    #     accu=val_acc,
                    #     target_accu=0.95,
                    #     log=print
                    # )

                # Get the final model train, validation, and test scores
                print(f"Getting final train, validation, and test accuracy after training.")
                train_actual, train_predicted, train_ptype_results = tnt.test(
                    model=ppnet_multi,
                    dataloader=trainloader,
                    class_specific=class_specific,
                    log=log
                )
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

                # Calculate and print the train, validation, and test metrics
                print("Calculating metrics")
                train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(ppnet, trainloader)
                print(f'\nTrain Accuracy: {train_accuracy}\n Train Precision: {train_precision}\n Train Recall: {train_recall}\n Train F1: {train_f1}')
                val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(ppnet, valloader)
                print(f'\nValidation Accuracy: {val_accuracy}\n Validation Precision: {val_precision}\n Validation Recall: {val_recall}\n Validation F1: {val_f1}')
                test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(ppnet, testloader)
                print(f'\nTest Accuracy: {test_accuracy}\n Test Precision: {test_precision}\n Test Recall: {test_recall}\n Test F1: {test_f1}')
                
                # # Compute metrics for validation set
                # print(f"Computing and storing results metrics")
                # val_m_f1 = metrics.f1_score(val_actual, val_predicted, average='macro')
                # val_m_recall = metrics.recall_score(val_actual, val_predicted, average='macro', zero_division=1)
                # val_acc = metrics.accuracy_score(val_actual, val_predicted)
                # val_m_precision = metrics.precision_score(val_actual, val_predicted, average='macro', zero_division=1)
                # val_w_precision = metrics.precision_score(val_actual, val_predicted, average='weighted', zero_division=1)
                # val_w_recall = metrics.recall_score(val_actual, val_predicted, average='weighted', zero_division=1)
                # val_w_f1 = metrics.f1_score(val_actual, val_predicted, average='weighted')
                # val_balanced_acc = metrics.balanced_accuracy_score(val_actual, val_predicted)

                # # Compute metrics for test set
                # test_m_f1 = metrics.f1_score(test_actual, test_predicted, average='macro')
                # test_m_recall = metrics.recall_score(test_actual, test_predicted, average='macro', zero_division=1)
                # test_acc = metrics.accuracy_score(test_actual, test_predicted)
                # test_m_precision = metrics.precision_score(test_actual, test_predicted, average='macro', zero_division=1)
                # test_w_precision = metrics.precision_score(test_actual, test_predicted, average='weighted', zero_division=1)
                # test_w_recall = metrics.recall_score(test_actual, test_predicted, average='weighted', zero_division=1)
                # test_w_f1 = metrics.f1_score(test_actual, test_predicted, average='weighted')
                # test_bal_acc = metrics.balanced_accuracy_score(test_actual, test_predicted)

                # print(f"Final val macro f1-score: {val_m_f1}")
                # print(f"Final val micro accuracy: {val_acc}")
                # print(f"Final test macro f1-score: {test_m_f1}")
                # print(f"Final test micro accuracy: {test_acc}")

                # print(f"Final validation:")
                # print(f"\tCluster: {val_ptype_results['cluster']}")
                # print(f"\tSeparation: {val_ptype_results['separation']}")
                # # print(f"Final val prototype results: {val_ptype_results}")
                # print(f"Final test:")
                # print(f"\tCluster: {test_ptype_results['cluster']}")
                # print(f"\tSeparation: {test_ptype_results['separation']}")
                # # print(f"Final test prototype results: {test_ptype_results}")

                # results = {
                #     # Results
                #     'val_macro_f1-score': val_m_f1,
                #     'val_macro_recall': val_m_recall, 
                #     'val_micro_accuracy': val_acc,
                #     'val_macro_precision': val_m_precision,
                #     'val_weighted_precision': val_w_precision, 
                #     'val_weighted_recall': val_w_recall,
                #     'val_weighted_f1-score': val_w_f1,
                #     'val_balanced_accuracy': val_balanced_acc,
                #     'test_macro_f1-score': test_m_f1,
                #     'test_macro_recall': test_m_recall,
                #     'test_micro_accuracy': test_acc,
                #     'test_macro_precision': test_m_precision,
                #     'test_weighted_precision': test_w_precision, 
                #     'test_weighted_recall': test_w_recall,
                #     'test_weighted_f1-score': test_w_f1,
                #     'test_balanced_accuracy': test_bal_acc,
                #     'epochs_taken': epoch+1,
                #     'test_time': test_ptype_results['time'],
                #     # Prototype Results
                #     'val_cross_ent': val_ptype_results['cross_ent'],
                #     'val_cluster': val_ptype_results['cluster'],
                #     'val_separation': val_ptype_results['separation'],
                #     'val_avg_separation': val_ptype_results['avg_separation'],
                #     'val_p_avg_pair_dist': val_ptype_results['p_avg_pair_dist'],


                #     # Variable variables
                #     'num_ptypes_per_class': params['prototype_shape'][0] / config['num_classes'], # num_ptypes_per_class,
                #     'ptype_length': params['prototype_shape'][2], # ptype_length,
                #     'prototype_shape': params['prototype_shape'],
                #     'ptype_activation_fn': 'unused',
                #     'latent_weight': params['latent_weight'],
                #     # joint is not used currently
                #     'joint_features_lr': params['joint_optimizer_lrs']['features'],
                #     'joint_ptypes_lr': params['joint_optimizer_lrs']['prototype_vectors'],
                #     'p0_warm_ptype_lr': params['p0_warm_ptype_lr'],
                #     'p1_warm_ptype_lr': params['p1_warm_ptype_lr'],
                #     'p2_warm_ptype_lr': params['p2_warm_ptype_lr'],
                #     'p3_warm_ptype_lr': params['p3_warm_ptype_lr'],
                #     'p1_last_layer_lr': params['p1_last_layer_lr'],
                #     'p2_last_layer_lr': params['p2_last_layer_lr'],
                #     'p3_last_layer_lr': params['p3_last_layer_lr'],
                #     'p4_last_layer_lr': params['p4_last_layer_lr'],
                #     'joint_weight_decay': params['joint_weight_decay'],
                #     'joint_lr_step_size': params['joint_lr_step_size'],
                #     'warm_lr_step_size': params['warm_lr_step_size'],
                #     'cross_entropy_weight': params['coefs']['crs_ent'],
                #     'cluster_weight': params['coefs']['clst'],
                #     'separation_weight': params['coefs']['sep'],
                #     'l1_weight': params['coefs']['l1'],
                #     'num_warm_epochs': params['num_warm_epochs'],
                #     'push_gap': params['push_gap'],
                #     'push_start': params['push_start'],
                #     'num_pushes': params['num_pushes'],
                #     'last_layer_iterations': params['last_layer_iterations'],

                #     # Static Variables
                #     'seq_count_thresh': config['seq_count_thresh'],           
                #     'trainRandomInsertions': config['trainRandomInsertions'],  
                #     'trainRandomDeletions': config['trainRandomDeletions'],  
                #     'trainMutationRate': config['trainMutationRate'],      
                #     'oversample': config['oversample'],               
                #     'encoding_mode': config['encoding_mode'],    
                #     'push_encoding_mode': config['push_encoding_mode'],  
                #     'applying_on_raw_data': config['applying_on_raw_data'],
                #     'load_existing_train_test': config['load_existing_train_test'], 
                #     'train_batch_size': config['train_batch_size'], 
                #     'test_batch_size': config['test_batch_size'],
                #     'num_classes': config['num_classes'],
                #     'seq_target_length': config['seq_target_length'],   
                #     'addTagAndPrimer': config['addTagAndPrimer'],
                #     'addRevComplements': config['addRevComplements'],
                #     'val_portion_of_train': val_portion,
                #     # 'patience': early_stopper.patience,
                #     # 'min_pc_improvement': early_stopper.min_pct_improvement,
                # }
                # utils.update_results(
                #     results,
                #     compare_cols='ppn',
                #     model=ppnet_multi,
                #     filename='search_for_warm_improvement_3_18_24.csv',
                #     save_model_dir = None
                #     # save_model_dir='saved_ppn_models'
                # )
                break # for early stopping

            elif epoch >= params['push_start'] and (epoch - params['push_start']) % params['push_gap'] == 0:
                print(f"Push epoch at epoch {epoch+1}.\n"
                    f"Final validation accuracy before push {pushes_completed + 1}: {val_acc}")
                print(f"Push number {pushes_completed + 1}")
                push_prototypes(
                    pushloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    preprocess_input_function=None, # normalize if needed
                    root_dir_for_saving_prototypes=None,#'./saved_prototypes', # if not None, prototypes will be saved here # Note: previously seq_dir
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    log=log,
                    sanity_check=True
                )
                pushes_completed += 1

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
                print(f"(Directly after push {pushes_completed}) Val acc at iteration 0: {val_acc}", flush=flush)

                # After pushing, retrain the last layer to produce good results again.
                tnt.last_only(model=ppnet_multi, log=log)
                print(f"Retraining last layer: ")

                # Print out the last layer lr
                for param_group in last_layer_optimizer.param_groups:
                        print(f"Last layer lr: {param_group['lr']}")

                for i in range(last_layer_iterations):
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

                    # conditionally_save_model(
                    #     ppnet,
                    #     'saved_ppn_models',
                    #     arr_job_id=str(args.arr_job_id),
                    #     comb_num=str(args.comb_num),
                    #     accu=val_acc,
                    #     target_accu=0.95,
                    #     log=print
                    # )
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
                for param_group in joint_optimizer.param_groups:
                    print(f"Joint optimizer lr: {param_group['lr']}")
                # print(f"Prototype results: {ptype_results}")
            

        # end of training and testing for given model
        # save the model results
        val_accs.append(val_accuracy)
        test_accs.append(test_accuracy)
        new_model_path = os.path.join('saved_ppn_models', (str(args.arr_job_id) + '_' + str(args.comb_num) + '_-1.pth'))
        torch.save(obj=ppnet_multi, f=new_model_path)
        
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
print(f"Test accuracies: {val_accs}")
print(f"Test accuracies: {test_accs}")
print(f"Over {num_trials} trials for all {combos} models evaluated in the grid search, got average validation accuracy: {np.mean(val_accs)}, standard deviation: {np.std(val_accs)}") 
print(f"Over {num_trials} trials for all {combos} models evaluated in the grid search, got average testing accuracy: {np.mean(test_accs)}, standard deviation: {np.std(test_accs)}") 

print(f"Finished search.")
