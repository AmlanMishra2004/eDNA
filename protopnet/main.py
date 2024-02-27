import os
import shutil

import itertools
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from sklearn import metrics
import warnings

import argparse
import re
import random
import pandas as pd
import torch.nn as nn

from helpers import makedir
from push import push_prototypes
import train_and_test as tnt
from preprocess import mean, std, preprocess_input_function
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

normalize = transforms.Normalize(mean=mean, std=std) # not used in this file or any other?

# Jon says: we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)

early_stopper = utils.EarlyStopping(
    patience=7,
    min_pct_improvement=1,#1,#3, # previously 20 epochs, 0.1% (for backbone)
    verbose=False
)

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
orig_train = pd.concat([X_train, y_train], axis=1)
if config['oversample']:
    train = utils.oversample_underrepresented_species(
        orig_train,
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
    early_stopper.reset()

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
    #     'weight_decay':             [0.065], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
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
    #     'last_layer_optimizer_lr':  [0.00065], #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
    #     'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
    #     'push_epochs_gap':          [10, 12, 14], # 17 #random.randint(10, 20)# 1_000_000 # not set
    #     'push_start':               [10, 15, 20, 25, 30], #25 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
    #     # BELOW IS UNUSED
    #     'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
    #     'joint_optimizer_lrs': [{ # learning rates for the different stages
    #         'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
    #         'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
    #     }]
    # }

    # search 2/27/24
    # These two are also hyperparameters. Feel free to add more values to try.
    num_ptypes_per_class = [2] #random.randint(1, 3) # not set
    ptype_length = [25] #random.choice([i for i in range(3, 30, 2)]) # not set, must be ODD
    hyperparameters = {    
        # comments after the line indicate jon's original settings
        # if the settings were not applicable, I write "not set".

        'prototype_shape':          [tuple(shape) for shape in [[config['num_classes']*ptypes, num_latent_channels+8, length] for ptypes in num_ptypes_per_class for length in ptype_length]], # not set
        'latent_weight':            [0.9], #random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 0.8
        'weight_decay':             [0.065], #random.uniform(0, 0.01) # 0.001, large number penalizes large weights
        'gamma':                    [1], #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]) # 0.3
        'warm_lr_step_size':        [1_000_000], # try 14? #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
        'coefs': [{ # weighting of different training losses
            'crs_ent':              1,
            'clst':                 1*12*-0.8,
            'sep':                  1*30*0.08,
            'l1':                   1e-3,
        }],
        'warm_optimizer_lrs': [{
            'prototype_vectors':    0.0007, #random.uniform(0.0001, 0.001) # 4e-2
        }], 
        'last_layer_optimizer_lr':  [0.00065], #random.uniform(0.0001, 0.001) # jon: 0.02, sam's OG: 0.002
        'num_warm_epochs':          [1_000_000], # random.randint(0, 10) # not set
        'push_epochs_gap':          [14], # 17 #random.randint(10, 20)# 1_000_000 # not set
        'push_start':               [30], #25 #random.randint(20, 30) # 1_000_000 #random.randint(0, 10) # not set #10_000_000
        # BELOW IS UNUSED
        'joint_lr_step_size':       [-1], #random.randint(1, 20) # not set, 20 is arbitrary and may or may not be greater than the number of epochs
        'joint_optimizer_lrs': [{ # learning rates for the different stages
            'features':             -1,#random.uniform(0.0001, 0.01), # 0.003
            'prototype_vectors':    -1 #random.uniform(0.0001, 0.01) # 0.003
        }]
    }

    # Generate all combinations of hyperparameters
    combinations = list(itertools.product(*hyperparameters.values()))

    # Iterate through all combinations
    combos = len(combinations)
    if combos > 1:
        print(f"\n\nExploring {combos} hyperparameter combinations for grid search.\n")
    for combination in combinations:
        params = dict(zip(hyperparameters.keys(), combination))
        print(f"\nAttempting combination:")
        for key, value in params.items():
            print(f"{key}: {value}")
        print('\n\n')
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
        # ppnet_multi = torch.nn.DataParallel(ppnet) # uncomment?
        ppnet_multi = ppnet
        class_specific = True

        # define optimizer
        # weight decay is how much to penalize large weights in the network, 0-0.1
        joint_optimizer_specs = [
            {'params': ppnet.features.parameters(),
            'lr': params['joint_optimizer_lrs']['features'],
            'weight_decay': params['weight_decay']}, # bias are now also being regularized
            {'params': ppnet.prototype_vectors,
            'lr': params['joint_optimizer_lrs']['prototype_vectors']},
        ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        # after step-size epochs, the lr is multiplied by gamma
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=params['joint_lr_step_size'], gamma=params['gamma'])

        warm_optimizer_specs = [
            {'params': ppnet.prototype_vectors,
            'lr': params['warm_optimizer_lrs']['prototype_vectors']}
        ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        # after step-size epochs, the lr is multiplied by gamma
        warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(warm_optimizer, step_size=params['warm_lr_step_size'], gamma=params['gamma'])
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(),
                                    'lr': params['last_layer_optimizer_lr']}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        for epoch in range(30_000):
            
            if epoch == 0:
                for param_group in warm_optimizer.param_groups:
                    param_group['lr'] *= 10
            elif epoch == 10:
                for param_group in warm_optimizer.param_groups:
                    param_group['lr'] /= 20
            elif epoch == 20:
                for param_group in warm_optimizer.param_groups:
                    param_group['lr'] /= 20

            if epoch >= params['push_start'] and (epoch - params['push_start']) % params['push_epochs_gap'] == 0:
                # Push epoch
                push_prototypes(
                    pushloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    preprocess_input_function=None, # normalize if needed
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here # sam: previously seq_dir
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    log=log
                )
                # After pushing, retrain the last layer to produce good results again.
                tnt.last_only(model=ppnet_multi, log=log)
                print(f"Retraining last layer: ")
                # Set the lr to the original lr
                for param_group in last_layer_optimizer.param_groups:
                    param_group['lr'] = params['last_layer_optimizer_lr']
                for i in range(20):
                    if i == 0:
                        for param_group in last_layer_optimizer.param_groups:
                            param_group['lr'] *= 10
                    elif i == 10:
                        for param_group in last_layer_optimizer.param_groups:
                            param_group['lr'] /= 10
                    elif i == 15:
                        for param_group in last_layer_optimizer.param_groups:
                            param_group['lr'] /= 10
                    actual, pred, _ = tnt.train(
                        model=ppnet_multi,
                        dataloader=trainloader,
                        optimizer=last_layer_optimizer,
                        class_specific=class_specific,
                        coefs=params['coefs'],
                        log=log
                    )
                    acc = torch.mean(actual == pred)
                    print(f"Train acc at iteration {i}: acc")
            elif epoch < params['num_warm_epochs']:
                # train the prototypes without modifying the backbone
                tnt.warm_only(model=ppnet_multi, log=log)
                _, _, _ = tnt.train(
                    model=ppnet_multi,
                    dataloader=trainloader,
                    optimizer=warm_optimizer,
                    scheduler=warm_lr_scheduler,
                    class_specific=class_specific,
                    coefs=params['coefs'],
                    log=log
                )
            else:
                # train the prototypes and the backbone
                tnt.joint(model=ppnet_multi, log=log)
                _, _, _ = tnt.train(
                    model=ppnet_multi,
                    dataloader=trainloader,
                    optimizer=joint_optimizer,
                    scheduler=joint_lr_scheduler,
                    class_specific=class_specific,
                    coefs=params['coefs'],
                    log=log
                )   
            # print(f"Calculating validation accuracy for epoch")
            val_actual, val_predicted, val_ptype_results  = tnt.test(
                model=ppnet_multi,
                dataloader=valloader,
                class_specific=class_specific,
                log=log
            )

            # warnings.filterwarnings("ignore")
            val_acc = metrics.accuracy_score(val_actual, val_predicted)
            # warnings.filterwarnings("default")

            early_stopper(val_acc)
            print(f"Val acc at epoch {epoch}: {val_acc}")
            # if val_acc > 0.3:
            #     print(f"Val acc at epoch {epoch}: {val_acc}")
            # if early_stopper.stop:

            # peaked around epoch [38*, 58, 73]

            # if epoch >= 55: # 45, expected: 38
            #     # give up on testing the model
            #     print("giving up on achieving desired accuracy")
            #     break
            if epoch >= 25 or val_acc >= 0.99: 
                print(f"Early stopping after epoch {epoch+1}.\n"
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
                # After pushing, retrain the last layer to produce good results again.
                tnt.last_only(model=ppnet_multi, log=log)
                print(f"Retraining last layer")
                for i in range(20):
                    if i == 0:
                        for param_group in last_layer_optimizer.param_groups:
                            param_group['lr'] *= 10
                    elif i == 10:
                        for param_group in last_layer_optimizer.param_groups:
                            param_group['lr'] /= 10
                    elif i == 15:
                        for param_group in last_layer_optimizer.param_groups:
                            param_group['lr'] /= 10
                    actual, pred, _ = tnt.train(
                        model=ppnet_multi,
                        dataloader=trainloader,
                        optimizer=last_layer_optimizer,
                        class_specific=class_specific,
                        coefs=params['coefs'],
                        log=log
                    )
                    acc = torch.mean(actual == pred)
                    print(f"Train acc at iteration {i}: acc")

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
                    'num_ptypes_per_class': num_ptypes_per_class,
                    'ptype_length': ptype_length,
                    'prototype_shape': params['prototype_shape'],
                    'ptype_activation_fn': 'unused',
                    'latent_weight': params['latent_weight'],
                    # joint is not used currently
                    'joint_features_lr': params['joint_optimizer_lrs']['features'],
                    'joint_add_on_layers_lr': -1,
                    'joint_ptypes_lr': params['joint_optimizer_lrs']['prototype_vectors'],
                    # only warm is used
                    'warm_add_on_layers_lr': -1,
                    'warm_ptypes_lr': params['warm_optimizer_lrs']['prototype_vectors'],
                    'last_layer_optimizer_lr': params['last_layer_optimizer_lr'],
                    'weight_decay': params['weight_decay'],
                    'joint_lr_step_size': params['joint_lr_step_size'],
                    'warm_lr_step_size': params['warm_lr_step_size'],
                    'cross_entropy_weight': params['coefs']['crs_ent'],
                    'cluster_weight': params['coefs']['clst'],
                    'separation_weight': params['coefs']['sep'],
                    'l1_weight': params['coefs']['l1'],
                    'num_warm_epochs': params['num_warm_epochs'],
                    'push_epochs_gap': params['push_epochs_gap'],
                    'push_start': params['push_start'],

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
                    'patience': early_stopper.patience,
                    'min_pc_improvement': early_stopper.min_pct_improvement,
                }
                utils.update_results(
                    results,
                    compare_cols='ppn',
                    model=ppnet_multi,
                    filename='ppnresults_gridsearch_2_27_24.csv',
                    save_model_dir='saved_ppn_models'
                )
                break # for early stopping

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