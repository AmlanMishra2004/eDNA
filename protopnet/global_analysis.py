import torch
import torch.utils.data
import numpy as np
import pandas as pd

import re
import sys

import os
from log import create_logger
from helpers import makedir

import argparse

sys.path.append('..')
from dataset import Sequence_Data
import utils
from torch.utils.data import DataLoader

# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-prototypeind', nargs=1, type=int)

#parser.add_argument('-dataset', nargs=1, type=str, default='cub200')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]
prot_ind=args.prototypeind[0]

load_model_path = os.path.join(load_model_dir, load_model_name)
start_epoch_number = 259

# load the model
ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = ppnet
# ppnet_multi = torch.nn.DataParallel(ppnet)

save_dir= './test_global'
model_base_architecture = 'small_best_updated'
experiment_run = '/'.join([load_model_name])
save_analysis_path = os.path.join(save_dir, model_base_architecture,
                                  experiment_run, load_model_name)
makedir(save_analysis_path)
log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'global_analysis.log'))


prototype_shape = ppnet.prototype_shape
#max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True


# load the data
# must use unaugmented (original) dataset

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

sequence_length = 70


train = pd.read_csv(config['train_path'], sep=',')
test = pd.read_csv(config['test_path'], sep=',')
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
test_dataset = Sequence_Data(
    test[config['seq_col']],
    test[config['species_col']],
    insertions=[config['testRandomInsertions']],
    deletions=config['testRandomDeletions'],
    mutation_rate=config['testMutationRate'],
    encoding_mode=config['encoding_mode'],
    seq_len=config['seq_target_length']
)

print('load model from ' + load_model_path)
print('model base architecture: ' + model_base_architecture)
print('experiment run: ' + experiment_run + '\n\n\n')


class_specific = True
load_img_dir = os.path.join(load_model_dir, 'seq')

##### HELPER FUNCTIONS FOR PLOTTING
def save_prototype(fname, epoch, index):
    p_seq = np.load(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype_'+str(index)+'_original.npy'))
    #plt.axis('off')
    np.save(fname, p_seq)

def save_prototype_patch(fname, epoch, index):
    p_seq = np.load(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype_'+str(index)+'_patch.npy'))
    #plt.axis('off')
    np.save(fname, p_seq)

def save_test_seq_patch(fname, patch_start, patch_end, test_seq):
    test_seq = test_seq[0]
    if patch_start < 0:
        # Handle zero padding
        target_patch = test_seq[:, :patch_end] #.cpu().detach().numpy()
        zeros = np.zeros((test_seq.shape[0], -patch_start))
        target_patch = np.concatenate((zeros, target_patch), axis=-1)

    elif patch_end > test_seq.shape[-1]:
        # Handle zero padding
        target_patch = test_seq[:, patch_start:]#.cpu().detach().numpy()
        zeros = np.zeros((target_patch.shape[0], patch_end - test_seq.shape[-1]))
        target_patch = np.concatenate((target_patch, zeros), axis=-1)
    else:
        target_patch = test_seq[:, patch_start:patch_end]#.cpu().detach().numpy()
    np.save(fname, target_patch)

def save_test_seq(fname, test_seq):
    np.save(fname, test_seq)

def save_act_map(fname, act_map):
    np.save(fname, act_map)


prototype_img_identity = torch.argmax(ppnet.prototype_class_identity, dim=1)

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(start_epoch_number), 'prototype_'+str(prot_ind)+'_patch.npy'))


save_prototype(os.path.join(save_analysis_path,
                                'original_prototype.npy'),
                   start_epoch_number, prot_ind)
save_prototype_patch(os.path.join(save_analysis_path,
                                'prototype_patch.npy'),
                   start_epoch_number, prot_ind)
# load the test image and forward it through the network

#to loop over
activation_pattern_table=[]
proto_act={}
test_dataset_len=int(test_dataset.num_samples)
print('test_dataset_len: ', test_dataset_len)
arg_max_list=[]
for i in range(test_dataset_len):

    seq, label = test_dataset.__getitem__(i)
    test_sequence = seq
    if type(test_sequence) is str:
        test_sequence_numpy = np.expand_dims(test_dataset.sequence_to_array(test_sequence), axis=0)
    else:
        test_sequence_numpy = np.expand_dims(test_sequence, axis=0)

    sequence_test = torch.tensor(test_sequence_numpy).cuda()

    conv_output, prototype_activation_patterns = ppnet.push_forward(sequence_test)
    activation_pattern_table.append(prototype_activation_patterns[:, prot_ind].cpu().detach().numpy())
    # array_act, sorted_indices_act = torch.sort(prototype_activations[idx])    

    max_proto_act = np.max(prototype_activation_patterns[:, prot_ind].cpu().detach().numpy())
    arg_max_proto_act= list(np.unravel_index(np.argmax(prototype_activation_patterns[:, prot_ind].cpu().detach().numpy(), axis=None),
                                prototype_activation_patterns.shape))
    arg_max_list.append(arg_max_proto_act[2])
    proto_act[i]=max_proto_act

print(activation_pattern_table)
print(arg_max_list)
makedir(os.path.join(save_analysis_path, 'most_activated_patches_for_prot_'+str(prot_ind)))
print('protoact: ',proto_act)
print('protoact_len: ',len(proto_act))
proto_act_sorted=sorted(proto_act.items(), key=lambda item: item[1])
print('sorted: ', proto_act_sorted)



for i in range(1, 11):
    # print(proto_act_sorted[-i][0])
    seq, label = test_dataset.__getitem__(proto_act_sorted[-i][0])
    if type(seq) is str:
        test_sequence_numpy = np.expand_dims(train_dataset.sequence_to_array(seq), axis=0)
    else:
        test_sequence_numpy = np.expand_dims(seq, axis=0)

    save_test_seq(os.path.join(save_analysis_path, 'most_activated_patches_for_prot_'+str(prot_ind), 'top-{0}_original_test_seq_{1}.npy'.format(i, str(label))),
                                        test_sequence_numpy)
    save_act_map(os.path.join(save_analysis_path, 'most_activated_patches_for_prot_'+str(prot_ind), 'top-{0}_prototype_activation_map_{1}.npy'.format(i, str(proto_act_sorted[-i][0]))),
                    activation_pattern_table[proto_act_sorted[-i][0]])
    upsampling_factor = 2
    proto_h = prototype_shape[-1]
    prototype_layer_stride = 1
    patch_start = upsampling_factor * (arg_max_list[proto_act_sorted[-i][0]] * prototype_layer_stride - proto_h // 2)
    patch_end = upsampling_factor * (arg_max_list[proto_act_sorted[-i][0]] + proto_h // 2) + upsampling_factor
    save_test_seq_patch(os.path.join(save_analysis_path, 'most_activated_patches_for_prot_'+str(prot_ind),
                                'top-{}_activated_test_patch.npy'.format(i)),
                                patch_start, patch_end, test_sequence_numpy)
    log('prototype class: {}'.format(prototype_img_identity[prot_ind]))
    log('test seq index: {0}'.format(proto_act_sorted[-i][0]))
    log('test seq class identity: {0}'.format(label))
    log('activation value (similarity score): {0}'.format(proto_act_sorted[-i][1]))
    log('--------------------------------------------------------------')


# for j in range(ppnet.num_prototypes):
#     makedir(os.path.join(root_dir_for_saving_train_seq, str(j)))
#     makedir(os.path.join(root_dir_for_saving_test_seq, str(j)))
#     save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_train_seq, str(j),
#                                                              'prototype_in_original_pimg.png'),
#                                           epoch=start_epoch_number,
#                                           index=j,
#                                           bbox_height_start=prototype_info[j][1],
#                                           bbox_height_end=prototype_info[j][2],
#                                           bbox_width_start=prototype_info[j][3],
#                                           bbox_width_end=prototype_info[j][4],
#                                           color=(0, 255, 255))
#     save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_test_seq, str(j),
#                                                              'prototype_in_original_pimg.png'),
#                                           epoch=start_epoch_number,
#                                           index=j,
#                                           bbox_height_start=prototype_info[j][1],
#                                           bbox_height_end=prototype_info[j][2],
#                                           bbox_width_start=prototype_info[j][3],
#                                           bbox_width_end=prototype_info[j][4],
#                                           color=(0, 255, 255))

# k = 5

# find_nearest.find_k_nearest_patches_to_prototypes(
#         dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
#         prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
#         k=k+1,
#         preprocess_input_function=preprocess_input_function, # normalize if needed
#         full_save=True,
#         root_dir_for_saving_seq=root_dir_for_saving_train_seq,
#         log=print)

# find_nearest.find_k_nearest_patches_to_prototypes(
#         dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
#         prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
#         k=k,
#         preprocess_input_function=preprocess_input_function, # normalize if needed
#         full_save=True,
#         root_dir_for_saving_seq=root_dir_for_saving_test_seq,
#         log=print)
