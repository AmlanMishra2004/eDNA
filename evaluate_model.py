# (c) 2023 Sam Waggoner
# License: AGPLv3

"""Performs grid search on a model and prints results, or uses AutoKeras.

First, the main function loads in and preprocesses the train sequences from the
reference database. Then, the program takes one of two routes. The first option
is using AutoKeras to perform a model search, then print results and save the
search progress and the best model. This requires that the data be fully
processed beforehand, since there is no online augmentation for AutoKeras. The
second route is performing grid search on one or more models from models.py.
After grid search is run, for each model, results are printed to the terminal
and optionally saved to a file.
"""

from collections import defaultdict
import datetime
import os
import random
import sys
import time

from joblib import dump, load
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from tabulate import tabulate
from tqdm import tqdm

from dataset import Sequence_Data
import models
from torch.utils.data import DataLoader
import utils

random.seed(1327)

def evaluate(model, X_train, y_train, X_test, y_test, k_folds, k_iters, epochs,
             optimizer, loss_function, early_stopper, batch_size,
             confidence_threshold, config, track_fold_epochs,
             track_test_epochs):
    """Trains and evaluates a given model, then displays results.

    Trains the model using k-fold cross validation,recording the validation
    accuracy and loss for each epoch and fold. Then retrains on the entire
    train set, then evaluates on a test set. Prints accuracy, precision,
    recall, F-1 score, and AUC-ROC score, along with graphs. If you are
    performing grid search, you will want to comment out the graphing so that
    you do not have to exit the graphs before moving on to the next model. 

    Args:
        model (torch.nn.Model): A PyTorch classification model.
        X_train (pandas.core.series.Series): All training examples, in a format
            that works with whatever model was provided. If any models from
            models.py are used, every element must be a string.
        y_train (pandas.core.series.Series): All training labels numerically
            encoded.
        X_test (pandas.core.series.Series): All testing examples, in a format
            that works with whatever model was provided. If any models from
            models.py are used, every element must be a string.
        y_test (pandas.core.series.Series): All testing labels numerically
            encoded.
        k_folds (int): Number of folds used in k-fold cross validation.
        k_iters (int): Number of iterations to train and validate. Ex. if
            k_iters = 2, then will train and validate on the first two folds
            and skip the remaining folds. Must be <=k_folds. 
            If k_iters is <1, then the model will skip validation and only
            perform the final training and testing.
        epochs (int): The maximum number of epochs to train the model. If an
            early_stopper is provided, the model may train for fewer epochs.
        optimizer (torch.optim): The PyTorch optimizer to use for updating the
            model parameters during training.
        loss_function (torch.nn.modules.loss): The PyTorch loss function to use
            during training.
        early_stopper (utils.EarlyStopping): An instance of the utils.py 
            EarlyStoppingclass with a defined patience and min_pct_improvement.
        batch_size (int): The number of examples to run before calculating the
            gradient and updating the model parameters.
        confidence_threshold (float or NoneType): If the maximum predicted
            category has a confidence less than this threshold, then the
            example will be ignored and excluded from computation for results.
            If None, it will not enforce a threshold. Example: 0.7.
        config (dict): A dictionary containing many hyperparameters. For a full
            list, refer to the correct usage in __main__().
NEW        track_fold_epochs (bool): Whether or not to evaluate on the validation
            set after each epoch during training on a given fold in order to
            graph train vs. test learning curves.
NEW        track_test_epochs (bool): Whether or not to evaluate on the test set
            after each epoch during final evaluation (after cross-validation)
            in order to graph train vs. test learning curves.
    Returns:
        tuple: (acc, epoch), where acc is the accuracy on the test set 
        (a float between 0 and 1), and epoch is the number of epochs the model 
        was trained (an integer).
    """

    # For k_iters number of splits in the train data,
    #   oversample train folds, train
    #   validate on validation fold
    # Train on entire train data
    # Evaluate on test data
    # print averaged validation results, print test results

    # Holds the validation results for each fold after the network has trained.
    # For each list below, each element is a dict of {macro_avg, weighted_avg}. 
    # This is with the exception of balanced_acc and epochs_taken. Training
    # metrics are not recorded during k-fold CV. For model comparison, I use
    # weighted average recall and weighted average f1-score.
    # Example: fold_val_metrics['precision'][0]['weighted'] will get the
    # weighted average precision for the validation set of the first fold.

    fold_val_metrics = {
        'precision': [],    # TP / TP+FP
        'recall': [],       # TP / TP+FN     measures the proportion of actual positives that are correctly identified
        'f1': [],           # 2 * precision*recall / precision+recall
        'acc': [],          # TP+TN / TP+FP+TN+FN
        'balanced_acc': [], # avg. recall on each class
        'epochs_taken': []
    }

    # Stores the metrics below such that each list contains <k_iters>
    # elements and each element is a list containing an entry for each epoch
    # that was used. This length can vary because of early stopping.
    # Example: learn_val_accuracies[4][5] would get the validation accuracy
    # achieved on the fifth fold after six epochs of training.
    # Losses contain the average loss per example, since dataset length varies.

    if track_fold_epochs:
        # old approach: np.empty((epochs, k_iters)), and 
        # learn_fold_train_accuracies[epoch, fold] = train_acc
        learn_fold_train_accuracies = [[] for _ in range(k_iters)] 
        learn_fold_val_accuracies = [[] for _ in range(k_iters)]
        learn_fold_train_losses = [[] for _ in range(k_iters)]
        learn_fold_val_losses = [[] for _ in range(k_iters)]

    np.set_printoptions(threshold=sys.maxsize)

    # K-FOLD STRATIFIED CV

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1327)
    for fold, (train_indexes, val_indexes) in enumerate(
        skf.split(X_train, y_train)
    ):
        if fold >= k_iters:
            break

        print(f"Starting fold {fold + 1} ------------------------------------")

        train_dataset = Sequence_Data(
            X=X_train.iloc[train_indexes],
            y=y_train.iloc[train_indexes],
            insertions=config['trainRandomInsertions'],
            deletions=config['trainRandomDeletions'],
            mutation_rate=config['trainMutationRate'],
            encoding_mode=config['encoding_mode'],
            seq_len=config['seq_target_length'])
        # NEW: oversample train dataset
        val_dataset = Sequence_Data(
            X=X_train.iloc[val_indexes],
            y=y_train.iloc[val_indexes], 
            insertions=config['testRandomInsertions'],
            deletions=config['testRandomDeletions'],
            mutation_rate=config['testMutationRate'],
            encoding_mode=config['encoding_mode'],
            seq_len=config['seq_target_length'])
        trainloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True)
        valloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False)

        model.reset_params() # Ensure weights are cleared from the prev. fold.
        if early_stopper:
            early_stopper.reset()

        for epoch in tqdm(range(epochs)):
            
            # TRAINING
            model.train()  # Sets the model to training mode.
            train_correct = 0
            total_loss = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()  # Zeros gradients for all the weights.
                try:
                    outputs = model(inputs)  # Runs the inputs through the model.
                except:
                    # in case the model is incorrectly set up, ex. with not enough nodes in the linear layer
                    return
                loss = loss_function(outputs, labels)  # Computes the loss.
                loss.backward()  # Performs backward pass, updating weights.
                optimizer.step()  # Performs optimization.
                _, predicted = torch.max(outputs, 1)

                total_loss += loss                 
                train_correct += (predicted == labels).sum().item()

            if track_fold_epochs:
                train_acc = train_correct / len(trainloader.dataset)
                learn_fold_train_accuracies[fold].append(train_acc)
                avg_loss = total_loss / len(trainloader.dataset)
                learn_fold_train_losses[fold].append(avg_loss)
        
            # VALIDATION
            model.eval() # Sets the model to evaluation mode.
            val_correct = 0
            total_loss = 0
            count = 0
            with torch.no_grad():
                # Iterate over the validation data and generate predictions.
                for i, (inputs, labels) in enumerate(valloader, 0):
                    inputs, labels = inputs.cuda(), labels.cuda()

                    outputs = model(inputs)
                    max_probs, predicted = torch.max(outputs, 1)

                    if confidence_threshold:
                        high_confidence_mask = max_probs > confidence_threshold
                        confident_predictions = predicted[high_confidence_mask]
                        confident_labels = labels[high_confidence_mask]
                        if track_fold_epochs:
                            confident_loss = loss_function(confident_predictions,
                                                           confident_labels)
                            total_loss += confident_loss

                        count += len(confident_predictions)
                        comparison = confident_predictions == confident_labels
                        val_correct += comparison.sum().item()
                    elif not confidence_threshold:
                        val_correct += (predicted == labels).sum().item()
                        if track_fold_epochs:
                            loss = loss_function(outputs, labels)
                            total_loss += loss

            if confidence_threshold:
                if count == 0:
                    print("Warning: No predictions met the confidence "
                          "threshold. Setting validation accuracy to 1.00")
                    val_acc = 1
                else:
                    val_acc = val_correct/count
            elif not confidence_threshold:
                val_acc = val_correct/len(valloader.dataset)     
            
            if track_fold_epochs:
                learn_fold_val_accuracies[fold].append(val_acc)
                avg_loss = total_loss / len(valloader.dataset)
                learn_fold_val_losses[fold].append(avg_loss)
            
            if early_stopper:
                early_stopper(val_acc)
                if early_stopper.stop:
                    print(f"Early stopping\n")
                    print(f"Fold {fold+1}, Epoch {epoch+1}/{epochs}, "
                          f"Validation Accuracy: {val_acc*100}%\n")
                    if confidence_threshold:
                        fold_val_metrics = utils.add_metrics_to_dict(
                            confident_labels,
                            confident_predictions,
                            epoch,
                            val_acc,
                            fold_val_metrics
                        )
                    elif not confidence_threshold:
                        fold_val_metrics = utils.add_metrics_to_dict(
                            labels,
                            predicted,
                            epoch,
                            val_acc,
                            fold_val_metrics
                        )
                    break
            
            # If the model has trained for all of its epochs without triggering
            # early stopping, then record metrics just like in early stopping.
                
            if epoch+1 == epochs:
                print(f"Finished training for full number of epochs.\n")
                print(f"Fold {fold+1}, Epoch {epoch+1}/{epochs}, "
                      f"Validation Accuracy: {val_acc*100}%\n")
                if confidence_threshold:
                    fold_val_metrics = utils.add_metrics_to_dict(
                        confident_labels,
                        confident_predictions,
                        epoch+1,
                        val_acc,
                        fold_val_metrics
                    )
                elif not confidence_threshold:
                    fold_val_metrics = utils.add_metrics_to_dict(
                        labels,
                        predicted,
                        epoch+1,
                        val_acc,
                        fold_val_metrics
                    )
            
    # EVALUATION
            
    if track_test_epochs:
        learn_train_accuracies = [[] for _ in range(k_iters)] 
        learn_train_losses = [[] for _ in range(k_iters)] 
            
    # First, retrain on the entire train dataset.
    train_dataset = Sequence_Data(
        X=X_train,
        y=y_train,
        insertions=config['trainRandomInsertions'],
        deletions=config['trainRandomDeletions'],
        mutation_rate=config['trainMutationRate'],
        encoding_mode=config['encoding_mode'],
        seq_len=config['seq_target_length'])
    test_dataset = Sequence_Data(
        X=X_test,
        y=y_test, 
        insertions=config['testRandomInsertions'],
        deletions=config['testRandomDeletions'],
        mutation_rate=config['testMutationRate'],
        encoding_mode=config['encoding_mode'],
        seq_len=config['seq_target_length'])
    trainloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True)
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False)

    model.reset_params() # Ensure weights are cleared from previous training.
    if early_stopper:
        early_stopper.reset()

    

    if k_iters == 0:
        train_epochs = epochs
    else:
        epochs_taken = fold_val_metrics['epochs_taken']
        train_epochs = int(sum(epochs_taken) / len(epochs_taken))

    for epoch in tqdm(range(train_epochs)):
        
        # TRAINING
        model.train()  # Sets the model to training mode.
        train_correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()  # Zeros gradients for all the weights.
            outputs = model(inputs)  # Runs the inputs through the model.
            loss = loss_function(outputs, labels)  # Computes the loss.
            loss.backward()  # Performs backward pass, updating weights.
            optimizer.step()  # Performs optimization.
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / len(trainloader.dataset)

        if track_test_epochs:
            learn_train_accuracies[fold].append(train_acc)
            learn_train_losses[fold].append(loss)

        # Since one can't use the test set for early stopping, simply train
        # for the average number of epochs taken during the k folds. Or, if you
        # want to risk overfitting, uncomment below.
            
        # if early_stopper:
        #         early_stopper(train_acc)
        #         if early_stopper.stop:
        #             print(f"Early stopping\n")
        #             break

    model.eval() # Set the model to evaluation mode.
    all_test_targets = []
    all_test_predictions = []
    all_test_outputs = []

    # TESTING
    start_time = time.time()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader, 0):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            max_probs, predicted = torch.max(outputs, 1)

            if confidence_threshold:
                confident_predictions = predicted[
                    max_probs > confidence_threshold
                ]
                confident_labels = labels[
                    max_probs > confidence_threshold
                ]
                # The total number of examples we are considering is no
                # longer the length of the trainloader.
                test_total += len(confident_predictions) 
                comparison = confident_predictions == confident_labels
                test_correct += comparison.sum().item()
                test_acc = test_correct/test_total
            else:
                test_correct += (predicted == targets).sum().item()
                test_acc = test_correct/len(testloader.dataset)

            # Record the actual values, predictions, & confidence percentages.
            all_test_targets.extend(targets.view(-1).tolist())
            all_test_predictions.extend(predicted.view(-1).tolist())
            all_test_outputs.extend(outputs.tolist())
    print(f"Test execution time: {time.time() - start_time} seconds")

    '''
    print training accuracy, other metrics
    print testing accuracy, other metrics

    if track_fold_epochs:
        train vs val graphs for each fold, for accuracy and loss
    if track_test_epochs:
        train accuracy and loss graphs
    training accuracy graph for final evaluation
    print some results
    save results to csv
    '''

    test_acc = accuracy_score(all_test_targets, all_test_predictions)
    test_m_precision = precision_score(all_test_targets,
                                     all_test_predictions,
                                     average="macro",
                                     zero_division=1)
    test_w_precision = precision_score(all_test_targets,
                                     all_test_predictions,
                                     average="weighted",
                                     zero_division=1)
    test_m_recall = recall_score(all_test_targets,
                               all_test_predictions,
                               average="macro",
                               zero_division=1)
    test_w_recall = recall_score(all_test_targets,
                               all_test_predictions,
                               average="weighted",
                               zero_division=1)
    test_m_f1 = f1_score(all_test_targets,
                       all_test_predictions,
                       average="macro",
                       zero_division=1)
    test_w_f1 = f1_score(all_test_targets,
                       all_test_predictions,
                       average="weighted",
                       zero_division=1)
    test_bal_acc = balanced_accuracy_score(all_test_targets,
                                           all_test_predictions)
    all_targets_one_hot = label_binarize(
        all_test_targets,
        classes=list(range(156))
    )
    roc_auc = roc_auc_score(
        all_targets_one_hot,
        all_test_outputs,
        multi_class='ovr',
        average='macro'
    )

    if k_iters > 1:
        avg_val_acc = sum(fold_val_metrics['acc']) / k_iters
        avg_val_m_precision = sum(x[0] for x in fold_val_metrics['precision']) / k_iters
        avg_val_w_precision = sum(x[1] for x in fold_val_metrics['precision']) / k_iters
        avg_val_m_recall = sum(x[0] for x in fold_val_metrics['recall']) / k_iters
        avg_val_w_recall = sum(x[1] for x in fold_val_metrics['recall']) / k_iters
        avg_val_m_f1 = sum(x[0] for x in fold_val_metrics['f1']) / k_iters
        avg_val_w_f1 = sum(x[1] for x in fold_val_metrics['f1']) / k_iters
        avg_val_balanced_acc = sum(fold_val_metrics['balanced_acc']) / k_iters
        avg_val_epochs_taken = sum(fold_val_metrics['epochs_taken']) / k_iters

        val_metrics = {
        'Metric': ['Micro Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1 Score', 'Weighted F1 Score','Balanced Accuracy', 'Epochs Taken'],
        'Value': [avg_val_acc, avg_val_m_precision, avg_val_m_recall, avg_val_m_f1, avg_val_w_f1, avg_val_balanced_acc, avg_val_epochs_taken]
        }
        print(f"Avg. of Validation Metrics over {k_folds} folds, {k_iters} iters.")
        print(tabulate(val_metrics, headers='keys', tablefmt='pretty'))

    # Test Metrics
    test_metrics = {
        'Metric': ['Micro Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1 Score', 'Weighted F1 Score', 'Balanced Accuracy', 'Macro OVR ROC-AUC'],
        'Value': [test_acc, test_m_precision, test_m_recall, test_m_f1, test_w_f1, test_bal_acc, roc_auc]
    }
    print("\nTest Metrics")
    print(tabulate(test_metrics, headers='keys', tablefmt='pretty'))

    results = {
        # For validation data, these metrics are the average over the k_iters
        # folds that were evaluated, where for each fold, the metric was likely
        # calculated with macro averaging. This means that the metric was
        # calculated for every example for each class, then the average class
        # scores were averaged to get the final number. Weighted averaging 
        # divides by the number of examples in each class.
        'val_macro_f1-score': avg_val_m_f1,
        'val_macro_recall': avg_val_m_recall, 
        'val_micro_accuracy': avg_val_acc,
        'val_macro_precision': avg_val_m_precision,
        'val_weighted_precision': avg_val_w_precision, 
        'val_weighted_recall': avg_val_w_recall,
        'val_weighted_f1-score': avg_val_w_f1,
        'val_balanced_accuracy': avg_val_balanced_acc,
        # 'val_macro_ovr_roc_auc_score': [], # not used

        'test_macro_f1-score': test_m_f1,
        'test_macro_recall': test_m_recall,
        'test_micro_accuracy': test_acc,
        'test_macro_precision': test_m_precision,
        'test_weighted_precision': test_w_precision, 
        'test_weighted_recall': test_w_recall,
        'test_weighted_f1-score': test_w_f1,
        'test_balanced_accuracy': test_bal_acc,
        'test_macro_ovr_roc_auc_score': roc_auc,

        # hyperparameters
        'learning_rate': optimizer.param_groups[0]['lr'],
        'loss_function': loss_function.__class__.__name__,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'betas': optimizer.param_groups[0]['betas'],
        'amsgrad': optimizer.param_groups[0]['amsgrad'],
        'weight_decay': optimizer.param_groups[0]['weight_decay'],
        'epochs_taken': train_epochs,
        'train_insertions': config['trainRandomInsertions'],
        'train_deletions': config['trainRandomDeletions'],
        'train_mutation_rate': config['trainMutationRate'], # xxx

        # preprocessing steps. differentiates rows.
        'model_name': model.name,
        'confidence_threshold': confidence_threshold,
        'seq_count_threshold': config['seq_count_thresh'],
        'seq_len': config['seq_target_length'],
        'test_insertions': config['testRandomInsertions'],
        'test_deletions': config['testRandomDeletions'],
        'test_mutation_rate': config['testMutationRate'],
        'tag_and_primer': config['addTagAndPrimer'],
        'reverse_complements': config['addRevComplements'],
        'k_folds': k_folds,
        'k_iters': k_iters,
        'encoding_mode': config['encoding_mode'], 
        'test_split': config['test_split'],
        'load_existing_train_test': config['load_existing_train_test']
    }
    # These results are for a single model with a single set of hyperparameters
    # and a single trial.
    utils.update_results(results, model)
    
    # # Graph results. NOTE: After reworking evaluate() on 12/26, I did not
    # # verify that this worked, since I haven't been needing it.
    # # Turn the lists of tensors into lists of lists on the CPU.
    # train_accuracies = utils.make_compatible_for_plotting(train_accuracies)
    # train_losses = utils.make_compatible_for_plotting(train_losses)
    # val_accuracies = utils.make_compatible_for_plotting(val_accuracies)
    # val_losses = utils.make_compatible_for_plotting(val_losses)
    # utils.graph_roc_curves(
    #     all_targets_one_hot,
    #     all_test_outputs,
    #     num_classes=156
    # )
    # # utils.graph_metric_over_time(train_accuracies, "Training", "Accuracy")
    # # utils.graph_metric_over_time(train_losses, "Training", "Loss")
    # # utils.graph_metric_over_time(val_accuracies, "Testing", "Accuracy")
    # # utils.graph_metric_over_time(val_losses, "Testing", "Loss")
    # utils.graph_train_vs_test(train_accuracies, val_accuracies, "Accuracy")
    # utils.graph_train_vs_test(train_losses, val_losses, "Loss")

    return test_acc, epoch+1

if __name__ == '__main__':
    """Loads in and preprocesses data, then trains and evaluates models.

    The following preprocessing steps are performed. All of the processing
    information is stored in the 'config' dict.
    1. load in training and testing data
    2. remove all unused sequences with <2 seq species
    3. (optionally) add tags and primers to every train sequence
    4. (optionally) add reverse complements of every train sequence
    5. train test stratified split
    6. oversample underrepresented train species
    7. in the tran and test Datasets, add...
        7a. random insertions per sequence
        7b. random deletions per sequence (either zeroing out a vector or
            deleting and adding a zeroed-vector to the end of the sequence)
        7c. mutation rate (random flips I think)
        7d. cap/fill to get 60 bp length
        7e. turn into a vector
    This function assumes that the csv file contains a header row, and the
    number of species' (number of categories) can be represented by 'long'.
    To save time, instead of repeating the offline preprocessing every time
    this file is run, if load_existing_train_test is set it will read in and
    use pre-existing train and test csv files.
    """

    # Overall Settings
    # If set, this will skip the preprocessing and read in an existing train
    # and test csv (that are presumably already processed). For my train and 
    # test file, all semutations have been added, so no need for online augmentation.Whether set or not, it still must be truncated/padded and turned intovectors.
    run_my_model = True
    run_autokeras = False
    run_baselines = False
    
    num_classes = 156

    print(f"Loading and processing data")

    config = {
        # IUPAC ambiguity codes represent if we are unsure if a base is one of
        # several options. For example, 'M' means it is either 'A' or 'C'.
        'iupac': {'a':'t', 't':'a', 'c':'g', 'g':'c',
                'r':'y', 'y':'r', 'k':'m', 'm':'k', 
                'b':'v', 'd':'h', 'h':'d', 'v':'b',
                's':'w', 'w':'s', 'n':'n', 'z':'z'},
        'data_path': './datasets/v4_combined_reference_sequences.csv',
        'train_path': './datasets/train.csv',
        'test_path': './datasets/test.csv',
        'sep': ';',                       # separator character in the csv file
        'species_col': 'species_cat',     # name of column containing species
        'seq_col': 'seq',                 # name of column containing sequences
        'seq_count_thresh': 2,            # ex. keep species with >1 sequences
        'test_split': 0.3,                # ex. .3 means 30% test, 70% train
        'trainRandomInsertions': [0,2],   # ex. between 0 and 2 per sequence
        'trainRandomDeletions': [0,2],    # ex. between 0 and 2 per sequence
        'trainMutationRate': 0.05,        # n*100% chance for a base to flip
        'encoding_mode': 'probability',   # 'probability' or 'random'
        # Whether or not applying on raw unlabeled data or "clean" ref db data.
        'applying_on_raw_data': False,
        # Whether or not to augment the test set.
        'augment_test_data': True,
        'load_existing_train_test': True, # use the same train/test split as Zurich, already saved in two different csv files
        'verbose': True
    }
    if config['applying_on_raw_data']:
        config['seq_target_length'] = 150
        config['addTagAndPrimer'] = True 
        config['addRevComplements'] = True
    elif not config['applying_on_raw_data']:
        config['seq_target_length'] = 60
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

    cols = ['species','family', 'genus', 'order']

    if not config['load_existing_train_test']:
        df = pd.read_csv(config['data_path'], sep=config['sep'])
        print(f"Original df shape: {df.shape}")
        utils.print_descriptive_stats(df, cols)
        # utils.plot_species_distribution(df, species_col)
        le = LabelEncoder()
        df[config['species_col']] = le.fit_transform(df[config['species_col']])
        dump(le, './datasets/label_encoder.joblib')
        df = utils.remove_species_with_too_few_sequences(
            df,
            config['species_col'],
            config['seq_count_thresh'],
            config['verbose']
        )
        if config['addTagAndPrimer']:
            df = utils.add_tag_and_primer(
                df,
                config['seq_col'],
                config['iupac'],
                'n'*10,
                'acaccgcccgtcactct',
                'cttccggtacacttaccatg',
                config['verbose']
            )
        if config['addRevComplements']:
            df = utils.add_reverse_complements(
                df,
                config['seq_col'],
                config['iupac'],
                config['verbose']
            )
        train, test = utils.stratified_split(
            df,
            config['species_col'],
            config['test_split']
        )
        train = utils.oversample_underrepresented_species(
            train,
            config['species_col'],
            config['verbose']
        )

        utils.print_descriptive_stats(df, cols)
        # utils.plot_species_distribution(df, species_col)

        # If you would like to verify that the train and test sets are the same
        # as in Zurich's paper, uncomment the lines below and compare files.
        # train.to_csv("/Users/Sam/OneDrive/Desktop/waggoner_train.csv")
        # test.to_csv("/Users/Sam/OneDrive/Desktop/waggoner_test.csv")
        # pause = input("PAUSE")

    elif config['load_existing_train_test']:
        train = pd.read_csv(config['train_path'], sep=',')
        test = pd.read_csv(config['test_path'], sep=',')

    if run_my_model:
        
        # If you want to see the format of the data will be fed into the model,
        # uncomment the code below.
        # torch.set_printoptions(threshold=sys.maxsize)
        # ex_trainloader = DataLoader(
        #                         train_dataset, 
        #                         batch_size=3,
        #                         shuffle=True)
        # ex_dataiter = iter(ex_trainloader)
        # ex_data, ex_labels = next(ex_dataiter)
        # print(f"Example data: \n{ex_data}\n")
        # print(f"Example label: \n{ex_labels}\n")
        # print("Shape of data: ", ex_data.shape)
        # print("Shape of labels: ", ex_labels.shape)

        start_time = time.time()

        cnn1 = models.CNN1(num_classes=num_classes)
        smallcnn2 = models.SmallCNN2(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        linear1 = models.Linear1(
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        linear2 = models.Linear2(
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        zurich = models.Zurich(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn1_1 = models.SmallCNN1_1(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn1_2 = models.SmallCNN1_2(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn2_1 = models.SmallCNN2_1(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn2_2 = models.SmallCNN2_2(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn2_3 = models.SmallCNN2_3(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn2_4 = models.SmallCNN2_4(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn2_6 = models.SmallCNN2_6(
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn3 = models.SmallCNN3(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )
        smallcnn3_1 = models.SmallCNN3_1(
            stride=1,
            in_width=config['seq_target_length'],
            num_classes=num_classes
        )

        # This list holds all of the models that will be trained and evaluated.
        models = [cnn1, zurich, smallcnn1_1, smallcnn1_2, smallcnn2, smallcnn2_1,
                  smallcnn2_2, smallcnn2_3, smallcnn2_4, smallcnn2_6, smallcnn3,
                  smallcnn3_1]

        for model in models:
            model.to('cuda')

        print(f"Evaluation for Personal Model(s):\n"
              f"{[f'{model.name}' for model in models]}")
        
        # The lines below set up the parameters for grid search.

        # num_trials sets the number of times each model with each set of
        # hyperparameters is run. Results are stored in a 2d list and averaged.
        num_trials = 3
        # learning_rates = [0.001]
        # learning_rates = [0.0005, 0.001]
        # learning_rates = [0.0005, 0.001, 0.003, 0.005]
        learning_rates = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        # learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 
        #                   0.01, 0.05]
        batch_size = 32
        epochs = 10_000
        # Zurich called the confidence_threshold 'binzarization threshold', and
        # used 0.9 for some of their evaluations. I do not compare my results
        # to those evaluations.
        confidence_threshold = None
        early_stopper = utils.EarlyStopping(
            patience=10,
            min_pct_improvement=0.5 # previously 20 epochs, 0.1%
        )
        k_folds = 5
        k_iters = 3 # Should be int [0 - k_folds]. Set to 0 to skip validation.

        # Grid search: Evaluates each model with a combination of
        # hyperparameters for a certain number of trials.
                
        for model in models:
            for idx, lr in enumerate(learning_rates):
                for trial in range(num_trials):
                    print(f"\n\nTraining Model {model.name}, Trial {trial+1}")

                    # To try to get the first batch to prove it is the same as Zurich
                    # first_batch = next(iter(trainloader))
                    # data,labels = first_batch
                    # data_path = "/Users/Sam/OneDrive/Desktop/my_first_batch_data.npy"
                    # labels_path = "/Users/Sam/OneDrive/Desktop/my_first_batch_labels.npy"
                    # np.save(data_path, data.numpy())
                    # np.save(labels_path, labels.numpy())

                    evaluate(
                        model,
                        X_train = train[config['seq_col']],
                        y_train = train[config['species_col']],
                        X_test = test[config['seq_col']],
                        y_test = test[config['species_col']],
                        k_folds = k_folds,
                        k_iters = k_iters,
                        epochs = epochs,
                        optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0, amsgrad=False),
                        loss_function = nn.CrossEntropyLoss(),
                        early_stopper = early_stopper,
                        batch_size = batch_size,
                        confidence_threshold = confidence_threshold,
                        config = config,
                        track_fold_epochs = False,
                        track_test_epochs = False)
                    
                    print(f"Total search runtime: {round((time.time() - start_time)/60,1)} minutes")
    
    if run_autokeras:

        # Import statements are included here because 1) Printing "Using
        # TensorFlow backend" to terminal may be misleading if AutoKeras is not
        # being used, and 2) it takes up to 10 seconds to import on my machine.

        import autokeras as ak
        from tensorflow.keras.models import load_model

# COMMENTED SINCE train, test already loaded right away
        # train = pd.read_csv(config['train_path'], sep=',')

        # # Since we are using this only for finding a model, we do not care
        # # about the test accuracy, we only care about finding the model that
        # # produces the best test accuracy then translating it to PyTorch and
        # # evaluating it. This is why we only test on the train set, and do not
        # # split the data.

        # train, test = train, train

        # train = utils.oversample_underrepresented_species(
        #     train,
        #     config['species_col'],
        #     config['verbose']
        # )
        # X_train, y_train = utils.encode_all_data(
        #     train.copy(),
        #     config['seq_target_length'],
        #     config['seq_col'],
        #     config['species_col'],
        #     config['encoding_mode'],
        #     True,
        #     "np",
        #     True,
        #     config['trainRandomInsertions'],
        #     config['trainRandomDeletions'],
        #     config['trainMutationRate']
        # )
        # X_test, y_test = utils.encode_all_data(
        #     test.copy(),
        #     config['seq_target_length'],
        #     config['seq_col'],
        #     config['species_col'],
        #     config['encoding_mode'],
        #     True,
        #     "np",
        #     True,
        #     config['testRandomInsertions'],
        #     config['testRandomDeletions'],
        #     config['testMutationRate']
        # )
        
        # print(f"X_train SHAPE: {X_train.shape}")
        # print(f"y_train SHAPE: {y_train.shape}")

        # Try AutoKeras using its image classifier.
        # Data shape is (11269, 1, 60, 4)

        X_train = train[config['seq_col']]
        y_train = train[config['species_col']]
        X_test = test[config['seq_col']]
        y_test = test[config['species_col']]

        pause = input("PAUSE")



        start_time = time.time()

        # Initialize and train the classifier
        clf = ak.ImageClassifier(overwrite=False, max_trials=300)
        clf.fit(X_train, y_train) # optionally, specify epochs=5

        # Export as a Keras model then save it.
        model = clf.export_model() 
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            model.save(f"saved_models/image_model_ak_{now}", save_format="tf")
        except Exception:
            model.save(f"saved_models/image_model_ak_{now}.h5")

        # Load the model back in for evaluations, and print its structure.
        loaded_model = load_model(
            f"saved_models/image_model_ak_{now}",
            custom_objects=ak.CUSTOM_OBJECTS
        )
        print(loaded_model.summary())

        results = loaded_model.predict(X_test)
        predicted_labels = np.argmax(results, axis=1)
        accuracy = np.mean(predicted_labels == y_test)
        print(f"Image test accuracy: {accuracy}")
        print(f"Image ak took {round((time.time() - start_time)/60,1)}"
              " seconds to run.")

        # I tried using AutoKeras StructuredDataClassifier to see if it would
        # work better than the ImageClassifier. However, it was unsuccessful.
        # Comment out the lines below if you want to try it.
        # Data shape should be (11269, 60, 4)

        # X_train_wo_height, y_train_wo_height = utils.encode_all_data(
        #     train.copy(), seq_target_length, seq_col, species_col,
        #     encoding_mode, False, "np")
        # X_test_wo_height, y_test_wo_height = utils.encode_all_data(
        #     test.copy(), seq_target_length, seq_col, species_col,
        #     encoding_mode, False, "np")
        # print(f"X_train_wo_height SHAPE: {X_train_wo_height.shape}")
        # print(f"y_train_wo_height SHAPE: {y_train_wo_height.shape}")

        # start_time = time.time()
        # clf = ak.StructuredDataClassifier(overwrite=True, max_trials=1)
        # # t = train_df[seq_col].to_frame()
        # # t = train_df[seq_col].apply(pd.Series)
        # t = pd.DataFrame(train_df[seq_col].to_numpy())
        # print(t[0].shape)
        # print(t[0][0].shape)
        # l = train_df[species_col]
        # clf.fit(t, l)
        # # clf.fit(X_train_wo_height, y_train_wo_height, epochs=5)

        # # Export as a Keras Model
        # model = clf.export_model()

        # # Save the model
        # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # try:
        #     model.save(f"saved_models/sdc_model_ak_{now}", save_format="tf")
        # except Exception:
        #     model.save(f"saved_models/sdc_model_ak_{now}.h5")

        # # Load the model
        # loaded_model = load_model(f"saved_models/sdc_model_ak_{now}",
        #   custom_objects=ak.CUSTOM_OBJECTS)
        # print(loaded_model.summary())

        # # Make predictions
        # results = loaded_model.predict(test_df[seq_col])

        # # Convert the predictions to the same type as your labels
        # predicted_labels = np.argmax(results, axis=1)

        # # Calculate the accuracy
        # accuracy = np.mean(predicted_labels == test_df[species_col]
        #   .to_numpy())
        # print(f"SDC test accuracy: {accuracy}")
        # print(f"SDC ak took {round((time.time() - start_time)/60,1)} "
        #   " minutes to run.")

    if run_baselines:

        # Note that for the baseline models, there is no need to 1) truncate or
        # pad the sequences to any specific length, or 2) convert the data to
        # 4D vectors. Sequences are kept as strings.

        from itertools import product
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.preprocessing import LabelEncoder

        np.set_printoptions(threshold=sys.maxsize)

        # read in unmutated train and test data

        train = pd.read_csv(config['train_path'], sep=',')
        test = pd.read_csv(config['test_path'], sep=',')

        mutation_options = {'a':'cgt', 'c':'agt', 'g':'act', 't':'acg'}
        # Defaultdict returns 'acgt' if a key is not present in the dictionary.
        mutation_options = defaultdict(lambda: 'acgt', mutation_options)

        train[config['seq_col']] = train[config['seq_col']].map(
            lambda x: utils.add_random_insertions(
                x,
                config['trainRandomInsertions']
            )
        )
        train[config['seq_col']] = train[config['seq_col']].map(
            lambda x: utils.apply_random_deletions(
                x,
                config['trainRandomDeletions']
            )
        )
        train[config['seq_col']] = train[config['seq_col']].map(
            lambda x: utils.apply_random_mutations(
                x,
                config['trainMutationRate'],
                mutation_options
            )
        )

        # # Pad or truncate to 60bp. 'z' padding may be turned to [0,0,0,0]
        # train[config['seq_col']] = train[config['seq_col']].apply(
        #     lambda seq: seq.ljust(config['seq_target_length'], 'z')[:config['seq_target_length']]
        # )

        # # Turn every base character into a vector.
        # train[config['seq_col']] = train[config['seq_col']].map(
        #     lambda x: utils.sequence_to_array(x, config['encoding_mode'])
        # )

        test[config['seq_col']] = test[config['seq_col']].map(
            lambda x: utils.add_random_insertions(
                x,
                config['testRandomInsertions']
            )
        )
        test[config['seq_col']] = test[config['seq_col']].map(
            lambda x: utils.apply_random_deletions(
                x,
                config['testRandomDeletions']
            )
        )
        test[config['seq_col']] = test[config['seq_col']].map(
            lambda x: utils.apply_random_mutations(
                x,
                config['testMutationRate'],
                mutation_options
            )
        )

        # # Pad or truncate to 60bp. 'z' padding may be turned to [0,0,0,0]
        # test[config['seq_col']] = test[config['seq_col']].apply(
        #     lambda seq: seq.ljust(config['seq_target_length'], 'z')[:config['seq_target_length']]
        # )

        # # Turn every base character into a vector.
        # test[config['seq_col']] = test[config['seq_col']].map(
        #     lambda x: utils.sequence_to_array(x, config['encoding_mode'])
        # )

        X_train =  train.loc[:,[config['seq_col']]].values
        y_train =  train.loc[:,[config['species_col']]].values
        X_test =  test.loc[:,[config['seq_col']]].values
        y_test =  test.loc[:,[config['species_col']]].values

        # Remove the inner lists for each element.
        X_train = X_train.ravel()
        X_test = X_test.ravel()
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"First two entries in X_train:\n{X_train[:2]}")
        print(f"First two entries in X_test:\n{X_test[:2]}")
        print(f"First two entries in y_train:\n{y_train[:2]}")
        print(f"First two entries in y_test:\n{y_test[:2]}")


        # Create the k-mer feature tables (X_train and X_test)

        try:
            print("Searching for pre-created k-mer feature tables...")
            # takes ~ 45 seconds
            ft_3 = np.load('./datasets/ft_3.npy')
            ft_5 = np.load('./datasets/ft_5.npy')
            ft_8 = np.load('./datasets/ft_8.npy')
            ft_10 = np.load('./datasets/ft_10.npy')

            ft_3_test = np.load('./datasets/ft_3_test.npy')
            ft_5_test = np.load('./datasets/ft_5_test.npy')
            ft_8_test = np.load('./datasets/ft_8_test.npy')
            ft_10_test = np.load('./datasets/ft_10_test.npy')

        except FileNotFoundError:
            print("Creating k-mer feature table for NB3 (expect <1 minute)...")
            ft_3 = utils.create_feature_table(X_train, 3)
            print("Creating k-mer feature table for NB5 (expect <1 minute)...")
            ft_5 = utils.create_feature_table(X_train, 5)
            print("Creating k-mer feature table for NB8 (expect ~1 minute)...")
            ft_8 = utils.create_feature_table(X_train, 8)
            # takes ~1 min. 65536        
            print("Creating k-mer feature table for NB10 (expect ~30 mins)...")
            ft_10 = utils.create_feature_table(X_train, 10)
            # takes ~17 mins. 1048576
            
            ft_3_test = utils.create_feature_table(X_test, 3)
            ft_5_test = utils.create_feature_table(X_test, 5)
            ft_8_test = utils.create_feature_table(X_test, 8)
            ft_10_test = utils.create_feature_table(X_test, 10)

            np.save('./datasets/ft_3.npy', ft_3)
            np.save('./datasets/ft_5.npy', ft_5)
            np.save('./datasets/ft_8.npy', ft_8)
            np.save('./datasets/ft_10.npy', ft_10)

            np.save('./datasets/ft_3_test.npy', ft_3_test)
            np.save('./datasets/ft_5_test.npy', ft_5_test)
            np.save('./datasets/ft_8_test.npy', ft_8_test)
            np.save('./datasets/ft_10_test.npy', ft_10_test)

        print(X_train[:5])
        print(ft_3[:5])
        print(f"Ft_3: {ft_3.shape}")
        print(f"Ft_5: {ft_5.shape}")
        print(f"Ft_8: {ft_8.shape}")
        print(f"Ft_10: {ft_10.shape}")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")

        # Create and train the baseline models

        try:
            print("Searching for pretrained naive bayes models...")
            NB_3 = load('./datasets/nb3.joblib')
            NB_5 = load('./datasets/nb5.joblib')
            NB_8 = load('./datasets/nb8.joblib')
            NB_10 = load('./datasets/nb10.joblib')
        except FileNotFoundError:
            print("Training NB3 (expect <1 min)...")
            # multinomial naive bayes
            NB_3 = MultinomialNB()
            NB_3.fit(ft_3, y_train)
            dump(NB_3, './datasets/nb3.joblib')
            print("Training NB5 (expect <1 min)...")
            NB_5 = MultinomialNB()
            NB_5.fit(ft_5, y_train)
            dump(NB_5, './datasets/nb5.joblib')
            print("Training NB8 (expect ~2 mins)...")
            NB_8 = MultinomialNB()
            NB_8.fit(ft_8, y_train)
            dump(NB_8, './datasets/nb8.joblib')
            NB_10 = MultinomialNB()
            NB_10.fit(ft_10, y_train)
            dump(NB_10, './datasets/nb10.joblib')

        try:
            print("Searching for pretrained random forest models...")
            # rf_15 = load('./datasets/rf15.joblib')
            rf_30 = load('./datasets/rf30.joblib')
            # rf_45 = load('./datasets/rf45.joblib')
        except FileNotFoundError:
            rf_15 = RandomForestClassifier(n_estimators=15, random_state=1327)
            rf_15.fit(X_train, y_train)
            dump(rf_15, './datasets/rf15.joblib')



        # Predict on test data and evaluate the baseline models
        # 0.00641025641 is random 1/156
        print("Evaluating models...")

        y_pred_nb3 = NB_3.predict(ft_3_test)
        accuracy = accuracy_score(y_test, y_pred_nb3)
        print(f"NB3 Accuracy: {accuracy}")
        print("NB3 Classification Report:")
        print(classification_report(y_test, y_pred_nb3, zero_division=1))
        print(f"NB3 Test accuracy: {NB_3.score(ft_3_test, y_test)}")
        print(f"NB3 Train accuracy: {NB_3.score(ft_3, y_train)}")

        y_pred_nb5 = NB_5.predict(ft_5_test)
        accuracy = accuracy_score(y_test, y_pred_nb5)
        print(f"NB5 Accuracy: {accuracy}")
        print("NB5 Classification Report:")
        print(classification_report(y_test, y_pred_nb5, zero_division=1))
        print(f"NB5 Test accuracy: {NB_5.score(ft_5_test, y_test)}")
        print(f"NB5 Train accuracy: {NB_5.score(ft_5, y_train)}")

        y_pred_nb8 = NB_8.predict(ft_8_test)
        accuracy = accuracy_score(y_test, y_pred_nb8)
        print(f"NB8 Accuracy: {accuracy}")
        print("NB8 Classification Report:")
        print(classification_report(y_test, y_pred_nb8, zero_division=1))
        print(f"NB8 Test accuracy: {NB_8.score(ft_8_test, y_test)}")
        print(f"NB8 Train accuracy: {NB_8.score(ft_8, y_train)}")

        y_pred_nb10 = NB_10.predict(ft_10_test)
        accuracy = accuracy_score(y_test, y_pred_nb10)
        print(f"NB10 Accuracy: {accuracy}")
        print("NB10 Classification Report:")
        print(classification_report(y_test, y_pred_nb10, zero_division=1))
        print(f"NB10 Test accuracy: {NB_10.score(ft_10_test, y_test)}")
        print(f"NB10 Train accuracy: {NB_10.score(ft_10, y_train)}")
