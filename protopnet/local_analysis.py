##### MODEL AND DATA LOADING
import argparse
import torch
import torch.utils.data
import numpy as np
import pandas as pd

import re

import os
import sys

from helpers import makedir
from log import create_logger

sys.path.append('..')
from dataset import Sequence_Data
import utils
from torch.utils.data import DataLoader



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

# /protopnet/local_results/epoch-n/[prototype_n_original.npy, prototype_n_activations.npy, prototype_n_patch.npy]

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-savedir', nargs=1, type=str)
# sequence should be the entire string representation of the target sequence
parser.add_argument('-targetrow', nargs=1, type=int)
parser.add_argument('-sequence', nargs=1, type=str, default='NA')
parser.add_argument('-seqclass', nargs=1, type=int, default=-1)
args = parser.parse_args()
print("\n\n\n\n\n\n\n\n\n")
print(f"gpuid: {args.gpuid[0]}")
print(f"modeldir: {args.modeldir[0]}")
print(f"model: {args.model[0]}")
print(f"savedir: {args.savedir[0]}")
print(f"targetrow: {args.targetrow[0]}")
print(f"sequence: {args.sequence[0]}")
print(f"seqclass: {args.seqclass[0]}")
print("\n\n\n\n\n\n\n\n\n")

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# specify the test image to be analyzed
save_dir = args.savedir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
test_sequence = args.sequence[0] #'Painted_Bunting_0081_15230.jpg'
test_sequence_label = args.seqclass[0] #15
target_row = args.targetrow[0] # The index of the test data sequence you want to analyze.

# load the model
load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/', now 1857326_0.9894.pth
load_model_name = args.model[0] #'10_18push0.7822.pth'
model_base_architecture = 'small_best_updated' #load_model_dir.split('/')[2]
experiment_run = '/'.join([load_model_name])
load_model_path = os.path.join(load_model_dir, load_model_name)
save_analysis_path = os.path.join(save_dir, model_base_architecture,
                                  experiment_run, load_model_name)
makedir(save_analysis_path)

# create the logger
log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

start_epoch_number = 259

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run + '\n\n\n')

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
# ppnet = torch.load(load_model_path, map_location=torch.device('cpu'))
# ppnet.to(torch.device('cuda'))

prototype_shape = ppnet.prototype_shape
class_specific = True
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
trainloader = DataLoader(
    train_dataset, 
    batch_size=config['train_batch_size'],
    shuffle=True
)
testloader = DataLoader(
    test_dataset,
    batch_size=config['test_batch_size'],
    shuffle=False
)

if target_row is not None:
    seq, label = test_dataset.__getitem__(target_row)
    test_sequence = seq
    test_sequence_label = label

load_img_dir = 'local_results'

##### SANITY CHECK
# confirm prototype class identity
prototype_img_identity = torch.argmax(ppnet.prototype_class_identity, dim=1)
log('Prototypes are chosen from ' + str(torch.max(prototype_img_identity)) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))
# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

##### HELPER FUNCTIONS FOR PLOTTING
def save_prototype(fname, epoch, index):
    file_to_load = os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype_'+str(index)+'_original.npy')
    log(f"File exists: {os.path.exists(file_to_load)}")
    p_seq = np.load(file_to_load)
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
        target_patch = test_seq[:, :patch_end].cpu().detach().numpy()
        zeros = np.zeros((test_seq.shape[0], -patch_start))
        target_patch = np.concatenate((zeros, target_patch), axis=-1)

    elif patch_end > test_seq.shape[-1]:
        # Handle zero padding
        target_patch = test_seq[:, patch_start:].cpu().detach().numpy()
        zeros = np.zeros((target_patch.shape[0], patch_end - test_seq.shape[-1]))
        target_patch = np.concatenate((target_patch, zeros), axis=-1)
    else:
        target_patch = test_seq[:, patch_start:patch_end].cpu().detach().numpy()
    np.save(fname, target_patch)

def save_test_seq(fname, test_seq):
    np.save(fname, test_seq)

def save_act_map(fname, act_map):
    np.save(fname, act_map)


# Load the test image and forward it through the network
if type(test_sequence) is str:
    test_sequence_numpy = np.expand_dims(test_dataset.sequence_to_array(test_sequence), axis=0)
else:
    test_sequence_numpy = np.expand_dims(test_sequence, axis=0)

sequence_test = torch.tensor(test_sequence_numpy).cuda()
labels_test = torch.tensor([test_sequence_label])

log(f"Test sequence: {sequence_test}")
log(f"Test label: {labels_test}")
logits, prototype_activations = ppnet(sequence_test)
conv_output, prototype_activation_patterns = ppnet.push_forward(sequence_test)

tables = []
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log('Predicted: ' + str(predicted_cls))
log('Actual: ' + str(correct_cls))

save_test_seq(os.path.join(save_analysis_path, 'original_seq.npy'),
                                     test_sequence_numpy)

##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
# (from any class)
makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

log('Most activated 10 prototypes of this image:')
array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
log(f"sorted_indices_act: {sorted_indices_act}")
log(f"array_act: {array_act}")
i = 1
i_completed = 0
while True: # for 10 iterations
    if i_completed == 11:
        break

    # Check if the prototype is saved. If it is not saved, skip it.
    file_to_load = os.path.join(
        load_img_dir,
        'epoch-'+str(start_epoch_number),
        'prototype_ '+ str(sorted_indices_act[-i].item()) + '_original.npy')
    saved_ptype_exists = os.path.exists(file_to_load)
    print(f"File {file_to_load} exists: {saved_ptype_exists}", flush=True)
    if not saved_ptype_exists:
        i += 1
        continue

    log('top {0} activated prototype for this image:'.format(i))
    # TODO: Fix all of this to save sequences instead of images
    log('Saving activation map')
    save_act_map(os.path.join(save_analysis_path, 'most_activated_prototypes', 'top-%d_prototype_activation_map.npy' % i),
                    prototype_activation_patterns[:, sorted_indices_act[-i].item()].cpu().detach().numpy())
    log('Saving prototype')
    log(f'Directory exists: {os.path.exists(save_analysis_path)}')
    save_prototype(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_prototype.npy' % i),
                   epoch=start_epoch_number, index=sorted_indices_act[-i].item())
    log('Saving prototype patch')
    save_prototype_patch(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_prototype_patch.npy' % i),
                   start_epoch_number, sorted_indices_act[-i].item())

    argmax_proto_act = \
        list(np.unravel_index(np.argmax(prototype_activation_patterns[:, sorted_indices_act[-i].item()].cpu().detach().numpy(), axis=None),
                                prototype_activation_patterns.shape))
    upsampling_factor = 2
    proto_h = prototype_shape[-1]
    prototype_layer_stride = 1
    patch_start = upsampling_factor * (argmax_proto_act[2] * prototype_layer_stride - proto_h // 2)
    patch_end = upsampling_factor * (argmax_proto_act[2] + proto_h // 2) + upsampling_factor
    print(argmax_proto_act[2], patch_start, patch_end)
    save_test_seq_patch(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_test_patch.npy' % i),
                                patch_start, patch_end, sequence_test)

    log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
    log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
    if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
        log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
    log('activation value (similarity score): {0}'.format(array_act[-i]))
    log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
    
    log('most highly activated patch of the chosen image by this prototype:')
    #plt.axis('off')
    
    log('most highly activated patch by this prototype shown in the original image:')
    
    print('--------------------------------------------------------------', flush=True)
    i_completed += 1

##### PROTOTYPES FROM TOP-k CLASSES
# (from the predicted class (which is not necessaryily the correct class))
k = 30
log('Prototypes from top-%d classes:' % k)
topk_logits, topk_classes = torch.topk(logits[idx], k=k)
for i,c in enumerate(topk_classes.detach().cpu().numpy()):
    makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

    log('top %d predicted class: %d' % (i+1, c))
    log('logit of the class: %f' % topk_logits[i])
    class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
    class_prototype_activations = prototype_activations[idx][class_prototype_indices]
    _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        prototype_index = class_prototype_indices[j]
        save_act_map(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'top-%d_prototype_activation_map.npy' % prototype_cnt),
                        prototype_activation_patterns[:, prototype_index].cpu().detach().numpy())
        save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'top-%d_activated_prototype.npy' % prototype_cnt),
                       start_epoch_number, prototype_index)
        save_prototype_patch(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'top-%d_activated_prototype_patch.npy' % prototype_cnt),
                       start_epoch_number, prototype_index)
                       
        log('prototype index: {0}'.format(prototype_index))
        log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
        if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
        log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))
        
        log('most highly activated patch of the chosen image by this prototype:')
        argmax_proto_act = \
            list(np.unravel_index(np.argmax(prototype_activation_patterns[:, prototype_index].cpu().detach().numpy(), axis=None),
                                    prototype_activation_patterns.shape))
        print(argmax_proto_act)
        proto_h = prototype_shape[-1]
        prototype_layer_stride = 1
        patch_start = upsampling_factor * (argmax_proto_act[2] * prototype_layer_stride - proto_h // 2)
        patch_end = upsampling_factor * (argmax_proto_act[2] + proto_h // 2) + upsampling_factor
        save_test_seq_patch(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'top-%d_activated_test_patch.npy' % prototype_cnt),
                                    patch_start, patch_end, sequence_test)

        print('--------------------------------------------------------------', flush=True)
        prototype_cnt += 1
    log('***************************************************************')

if predicted_cls == correct_cls:
    log('Prediction is correct.')
else:
    log('Prediction is wrong.')

logclose()

