##### MODEL AND DATA LOADING
import argparse
import torch
import torch.utils.data
import numpy as np
import pandas as pd

import os
import sys
import re

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
    'train_path': '../datasets/train_no_dup.csv',
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

# /protopnet/local_results/epoch-n/[prototype_n_original.npy, prototype_n_activations.npy, prototype_n_patch.npy]

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-experimentname', nargs=1, type=str)
parser.add_argument('-savedir', nargs=1, type=str)
# sequence should be the entire string representation of the target sequence
parser.add_argument('-targetrow', nargs=1, type=int)
parser.add_argument('-sequence', nargs=1, type=str, default='NA')
parser.add_argument('-trainortest', nargs=1, type=str)
parser.add_argument('-seqclass', nargs=1, type=int, default=[-1])
args = parser.parse_args()
print("\n\n\n\n\n\n\n\n\n")
print(f"gpuid: {args.gpuid[0]}")
print(f"modeldir: {args.modeldir[0]}")
print(f"model: {args.model[0]}")
print(f"experimentname: {args.experimentname[0]}")
print(f"savedir: {args.savedir[0]}")
print(f"targetrow: {args.targetrow[0]}")
print(f"sequence: {args.sequence[0]}")
print(f"trainortest: {args.trainortest[0]}")
print(f"seqclass: {args.seqclass[0]}")
print("\n\n\n\n\n\n\n\n\n")

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# specify the test image to be analyzed
save_dir = args.savedir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'

# The following two will be overridden if a valid target_row is provided
test_sequence = args.sequence[0] #'Painted_Bunting_0081_15230.jpg'
test_sequence_label = args.seqclass[0] #15

target_row = args.targetrow[0] # The index of the test data sequence you want to analyze.
train_or_test = args.trainortest[0] # Whether to grab the sequence from the train or test set

load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/', now 1857326_0.9894.pth
load_model_name = args.model[0] #'10_18push0.7822.pth'
experiment_name = args.experimentname[0] #'1892566_8_-1_latent_0.7' foldername to load the prototypes from
model_base_architecture = 'small_best_updated' #load_model_dir.split('/')[2]

load_ptype_dir = os.path.join('saved_prototypes', experiment_name) # ./saved_prototypes / 1878231_3_-1_latent_1_old /
load_model_path = os.path.join(load_model_dir, load_model_name) # ./saved_ppn_models / 1878231_3_-1.pth
save_analysis_path = os.path.join(save_dir, model_base_architecture,
                                  experiment_name, load_model_name) # "./local_results/test_local_seq_$IND / small_best_updated / 1892566_8_-1_latent_0.7 / 184702_4_0.95.pth /"

makedir(save_analysis_path)

# print(f"experiment_name: {experiment_name}")
# print(f"model_base_architecture: {model_base_architecture}")
# print(f"load_model_name: {load_model_name}")
# print(f"Load model dir: {load_model_dir}")

print(f"CWD: {os.getcwd()}")
print(f"prototypes are loaded from: {load_ptype_dir}")
print(f"model is loaded from: {load_model_path}")
print(f"local results are saved to: {save_analysis_path}\n\n\n")


# create the logger
log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

start_epoch_number = 214 # 259 # 1878231_3_-1.pth
# load_model_path = os.path.join(load_model_dir, load_model_name)
# start_epoch_number = int(re.search(r'\d+', load_model_name).group(0))

# load_model_path = './saved_ppn_models/1878231_3_-1.pth'
ppnet = torch.load(load_model_path, map_location=torch.device('cpu'))
# ppnet = ppnet.cuda()

# wait = input("PAUSE")

##### SANITY CHECK
# confirm prototype class identity
prototype_img_identity = torch.argmax(ppnet.prototype_class_identity, dim=1)
# log('Prototypes are chosen from ' + str(torch.max(prototype_img_identity)) + ' number of classes.')
# log('Their class identities are: ' + str(prototype_img_identity))
# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu()
# print(f"The maximum connection from each prototype to any node (class) in the classification layer is: \
    # {prototype_max_connection}")
if torch.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')
    print(f"{torch.sum(prototype_max_connection == prototype_img_identity)} out of \
        {ppnet.num_prototypes} prototypes belong identify most strongly with \
            their own class")
    
# wait = input("PAUSE")

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

if target_row is not None:
    if train_or_test == 'test':
        seq, label = test_dataset.__getitem__(target_row)
    elif train_or_test == 'train':
        seq, label = train_dataset.__getitem__(target_row)
    test_sequence = seq
    test_sequence_label = label

def array_to_str(arr):
    # print(f"arr.shape: {arr.shape}")
    res_str = ""
    for i in range(arr.shape[1]):
        if arr[0, i] == 1:
            res_str = res_str + 'A'
        elif arr[1, i] == 1:
            res_str = res_str + 'T'
        elif arr[2, i] == 1:
            res_str = res_str + 'C'
        elif arr[3, i] == 1:
            res_str = res_str + 'G'
        else:
            res_str = res_str + '_'
    return res_str

print(f"Array: {array_to_str(seq)}")
print(f"Label: {label}")

# pause = input("PAUSE")


##### HELPER FUNCTIONS FOR PLOTTING
def save_prototype(fname, epoch, index):
    file_to_load = os.path.join(load_ptype_dir, 'epoch-'+str(epoch), 'prototype_'+str(index)+'_original.npy')
    log(f"File exists: {os.path.exists(file_to_load)}")
    p_seq = np.load(file_to_load)
    #plt.axis('off')
    np.save(fname, p_seq)

def save_prototype_patch(fname, epoch, index):
    p_seq = np.load(os.path.join(load_ptype_dir, 'epoch-'+str(epoch), 'prototype_'+str(index)+'_patch.npy'))
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
    print(f"Saving the activation map below at {fname}")
    print(f"Activation map: {act_map}")
    np.save(fname, act_map)

def printinfo(varname, arr):
    print(f"{varname}: {arr}")
    print(f"{varname} shape: {arr.shape}")


# Load the test image and forward it through the network
# Add a dimension to the front 
if type(test_sequence) is str:
    test_sequence_numpy = np.expand_dims(utils.sequence_to_array(test_sequence, 'probability'), axis=0)
else:
    test_sequence_numpy = np.expand_dims(test_sequence, axis=0)

print(f"test_sequence_numpy.shape: {test_sequence_numpy.shape}") # (1, 4, 70)
print(f"test_sequence_label: {test_sequence_label}") # integer


sequence_test = torch.tensor(test_sequence_numpy)#.cuda() # (1, 4, 70)
labels_test = torch.tensor([test_sequence_label]) # (1)

save_test_seq(os.path.join(save_analysis_path, 'original_seq.npy'),
                                     test_sequence_numpy)


# log(f"Test sequence: {sequence_test}")
log(f"Test label: {labels_test}")
logits, prototype_activations = ppnet(sequence_test)
conv_output, prototype_activation_patterns = ppnet.push_forward(sequence_test)

# print(f"logits: {logits}")
print(f"logits.shape: {logits.shape}") # torch.Size([1, 156]) a bunch of -30 to -34 values
# print(f"prototype_activations: {prototype_activations}") 
print(f"prototype_activations.shape: {prototype_activations.shape}") # torch.Size([1, 468 (156*3)])
# print(f"conv_output: {conv_output}") # torch.Size([1, 520, 35])
print(f"conv_output.shape: {conv_output.shape}")
# print(f"prototype_activation_patterns: {prototype_activation_patterns}")
print(f"prototype_activation_patterns.shape: {prototype_activation_patterns.shape}") # torch.Size([1, 468 (156*3), 7]) 7 because 35 - ptype_length + 1 = 7

print(f"Prototype shape: {ppnet.prototype_shape}") # (468, 520, 29)
# printinfo("ppnet.prototype_vectors", ppnet.prototype_vectors)
# print(f"ppnet.prototype_vectors: {ppnet.prototype_vectors}") # 
# print(f"ppnet.prototype_vectors.shape: {ppnet.prototype_vectors.shape}") # 


# <# test images> long, where each entry has the index of the test image and a tuple (predicted class, actual class)
tables = []
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log('Predicted: ' + str(predicted_cls))
log('Actual: ' + str(correct_cls))

# pause = input("PAUSE")

##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
# (from any class)
makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

log('\n\n\nMost activated 10 prototypes of this image:\n')
sorted_array_acts, sorted_indices_act = torch.sort(prototype_activations[idx])
# log(f"sorted_indices_act: {sorted_indices_act}")
# log(f"sorted_array_acts: {sorted_array_acts}")

i = 1
i_completed = 0
while True: # for 10 iterations
    if i == 11:
        break
    if i_completed == 11:
        break

    # Check if the prototype is saved. If it is not saved, skip it.
    file_to_load = os.path.join(
        load_ptype_dir,
        'epoch-'+str(start_epoch_number),
        'prototype_'+ str(sorted_indices_act[-i].item()) + '_original.npy')
    saved_ptype_exists = os.path.exists(file_to_load)
    if not saved_ptype_exists:
        print(f"File {file_to_load} does not exist.", flush=True)
        i += 1
        continue

    log('top {0} activated prototype for this image:'.format(i))
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
    center_loc = argmax_proto_act[2]

    patch_start = center_loc * upsampling_factor
    patch_end = (center_loc + proto_h) * upsampling_factor
    # patch_start = upsampling_factor * (argmax_proto_act[2] * prototype_layer_stride - proto_h // 2)
    # patch_end = upsampling_factor * (argmax_proto_act[2] + proto_h // 2) + upsampling_factor

    print(argmax_proto_act[2], patch_start, patch_end)
    save_test_seq_patch(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_test_patch.npy' % i),
                                patch_start, patch_end, sequence_test)

    log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
    log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
    if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
        log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
    log('activation value (similarity score): {0}'.format(sorted_array_acts[-i]))
    log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
    
    # log('most highly activated patch of the chosen image by this prototype:')
    #plt.axis('off')
    
    # log('most highly activated patch by this prototype shown in the original image:')
    
    print('--------------------------------------------------------------', flush=True)
    i_completed += 1
    i += 1

log("\n\n\n\nFinished finding nearest 10 prototypes of all prototypes.")
log("Now, finding the top-activated prototypes for the top k classes\n\n\n\n")

# ./local_results/test_local_seq_$IND / small_best_updated / 1857326_0.9894.pth / 1857326_0.9894.pth

##### PROTOTYPES FROM TOP-k CLASSES
# (top-1 is the predicted class (which is not necessarily the correct class))
k = 3
log(f'\nPrototypes from top-{k} classes:\n')
topk_logits, topk_classes = torch.topk(logits[idx], k=k)
print(f"topk_logits: {topk_logits}", flush=True)
print(f"topk_classes: {topk_classes}", flush=True)
# For each of the top k predicted classes,
for i,c in enumerate(topk_classes.detach().cpu().numpy()):

    makedir(os.path.join(save_analysis_path, f'top-{i+1}_class_prototypes'))

    log('top %d predicted class: %d' % (i+1, c))
    log('logit of the class: %f' % topk_logits[i])
    class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
    class_prototype_activations = prototype_activations[idx][class_prototype_indices]
    _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

    prototype_cnt = 1
    log(f"sorted_indices_cls_act: {sorted_indices_cls_act}")
    # For each of the activated prototypes for the current class,
    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        prototype_index = class_prototype_indices[j]

        # # Check if the prototype is saved. If it is not saved, skip it.
        # file_to_load = os.path.join(
        #     save_analysis_path,
        #     'top-%d_class_prototypes' % (i+1),
        #     'top-%d_activated_prototype.npy' % prototype_cnt)
        # saved_ptype_exists = os.path.exists(file_to_load)
        # if not saved_ptype_exists:
        #     log(f"The prototype for this class does not exist: {file_to_load}")
        #     continue

        save_act_map(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'top-%d_prototype_activation_map.npy' % prototype_cnt),
                        prototype_activation_patterns[:, prototype_index].cpu().detach().numpy())

        save_prototype(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'top-%d_activated_prototype.npy' % prototype_cnt),
                       epoch=start_epoch_number, index=prototype_index)
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

