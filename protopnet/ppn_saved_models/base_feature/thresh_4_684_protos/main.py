import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
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

# allow the user to call the program with command line gpu argument
# ex. python3 main.py -gpuid=0,1,2,3
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# import all of the "hyperparameters" and settings
from settings import base_architecture, sequence_length, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run, latent_weight, \
                    use_aug, flip_count, del_prob

print("use_aug: ", use_aug)
print("flip_count: ", flip_count)
print("del_prob: ", del_prob)
base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './ppn_saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'base_model.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
seq_dir = os.path.join(model_dir, 'seq')
makedir(seq_dir)
weight_matrix_filename = 'outputL_weights'
prototype_seq_filename_prefix = 'prototype-seq'
prototype_self_act_filename_prefix = 'prototype-self-act'
# proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size, \
                     oversample, config

normalize = transforms.Normalize(mean=mean, std=std)

train = pd.read_csv(config['train_path'], sep=',')
test = pd.read_csv(config['test_path'], sep=',')

X_train, X_val, y_train, y_val = train_test_split(
    train[config['seq_col']], # X
    train[config['species_col']], # y
    test_size=0.2,
    random_state=42,
    stratify = train[config['species_col']]
)
orig_train = pd.concat([X_train, y_train], axis=1)
if oversample:
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
    batch_size=config['batch_size'],
    shuffle=True
)
valloader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)
testloader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
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
    encoding_mode=config['encoding_mode'],
    seq_len=config['seq_target_length']
)
pushloader = DataLoader(
    push_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(trainloader.dataset)))
log('push set size: {0}'.format(len(pushloader.dataset)))
log('test set size: {0}'.format(len(testloader.dataset)))
log('batch size: {0}'.format(train_batch_size))

print("prototype_shape")
print(prototype_shape)

# construct the model
model_path = "../saved_models/best_model_20240101_035108.pt"
best_12_31 = models.Best_12_31()
best_12_31.load_state_dict(torch.load(model_path))
# remove the linear layer
backbone = best_12_31
backbone.linear_layer = nn.Identity()
ppnet = ppn.construct_PPNet(
    features=backbone, # CHANGE, should have pre-loaded weights
    pretrained=True, sequence_length=sequence_length,
    prototype_shape=prototype_shape,
    num_classes=num_classes,
    prototype_activation_function=prototype_activation_function,
    add_on_layers_type=add_on_layers_type,
    latent_weight=latent_weight
)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
# ppnet_multi = torch.nn.DataParallel(ppnet)
ppnet_multi = ppnet
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
gamma = 0.3
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=gamma)

print("Trying gamma {}".format(gamma))
print("joint_lr_step_size")
print(joint_lr_step_size)
print("joint_optimizer_lrs")
print(joint_optimizer_lrs)

# optimize the weights for the linear layer i think?
from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
print("warm_optimizer_lrs")
print(warm_optimizer_lrs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs
print("coefs: ", coefs)

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model. based on which epoch
log('start training')
max_accu = 0
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, _ = tnt.train(model=ppnet_multi, dataloader=trainloader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        _, _ = tnt.train(model=ppnet_multi, dataloader=trainloader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
        joint_lr_scheduler.step()

    accu, cm = tnt.test(model=ppnet_multi, dataloader=testloader,
                    class_specific=class_specific, log=log)
    if accu > max_accu:
        max_accu = accu
        log('new max accuracy: \t{0}%'.format(max_accu * 100))
    #, cm
    #save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
    #                            target_accu=0.82, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push_prototypes(
            pushloader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            preprocess_input_function=None, # normalize if needed
            root_dir_for_saving_prototypes=seq_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            log=log)
        accu, cm = tnt.test(model=ppnet_multi, dataloader=testloader,
                        class_specific=class_specific, log=log)
        # UNCOMMENT for LOCAL ANALYSIS i think
        # save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
        #                              target_accu=0.70, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _, _ = tnt.train(model=ppnet_multi, dataloader=trainloader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu, cm = tnt.test(model=ppnet_multi, dataloader=testloader,
                                class_specific=class_specific, log=log)
                save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                           target_accu=0.70, log=log)

    if epoch>num_train_epochs-50:
        log('\tconfusion matrix: \t\t\n{0}'.format(cm))

  
logclose()
