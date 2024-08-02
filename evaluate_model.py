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
print("Importing libraries", flush=True)
from collections import defaultdict
from datetime import datetime
import os
import random
import sys
import time
import warnings

from joblib import dump, load
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from tabulate import tabulate
from tqdm import tqdm

from dataset import Sequence_Data
import models
from torch.utils.data import DataLoader
import utils


# random.seed(1327)

def evaluate(model, train, test, k_folds, k_iters, epochs, oversample, 
             optimizer, loss_function, early_stopper, batch_size,
             confidence_threshold, config, track_fold_epochs,
             track_test_epochs, num_classes, results_file_name='results.csv'):
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
        track_fold_epochs (bool): Whether or not to evaluate on the validation
            set after each epoch during training on a given fold in order to
            graph train vs. test learning curves.
        track_test_epochs (bool): Whether or not to evaluate on the test set
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
    warnings.filterwarnings("ignore", "y_pred contains classes not in y_true")

    # K-FOLD STRATIFIED CV

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1327)
    for fold, (train_indexes, val_indexes) in enumerate(
        skf.split(train[config['seq_col']], train[config['species_col']])
    ):
        if fold >= k_iters:
            break

        print(f"Starting fold {fold + 1}")

        train_fold = train.iloc[train_indexes]
        val_fold = train.iloc[val_indexes]

        if oversample:
            train_fold = utils.oversample_underrepresented_species(
                train_fold,
                config['species_col'],
                config['verbose']
            )
        train_dataset = Sequence_Data(
            X=train_fold[config['seq_col']],
            y=train_fold[config['species_col']],
            insertions=config['trainRandomInsertions'],
            deletions=config['trainRandomDeletions'],
            mutation_rate=config['trainMutationRate'],
            encoding_mode=config['encoding_mode'],
            seq_len=config['seq_target_length'])
        val_dataset = Sequence_Data(
            X=val_fold[config['seq_col']],
            y=val_fold[config['species_col']],
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
        

        # UNCOMMENT THIS IF YOU JUST WANT TO EVALUATE A MODEL

        # test_dataset = Sequence_Data(
        #     X=test[config['seq_col']],
        #     y=test[config['species_col']], 
        #     insertions=config['testRandomInsertions'],
        #     deletions=config['testRandomDeletions'],
        #     mutation_rate=config['testMutationRate'],
        #     encoding_mode=config['encoding_mode'],
        #     seq_len=config['seq_target_length'])
        # testloader = DataLoader(
        #     test_dataset,
        #     batch_size=batch_size,
        #     shuffle=False)
        # def calculate_accuracy(dataloader):
        #     correct = 0
        #     total = 0
        #     with torch.no_grad():
        #         for data in dataloader:
        #             inputs, labels = data[0].to('cuda'), data[1].to('cuda')
        #             outputs = model(inputs) # returns (logits, max_similarities)
        #             # print(outputs)
        #             # pause = input("Pause")
        #             _, predicted = torch.max(outputs[0], 1)
        #             total += labels.size(0)
        #             correct += (predicted == labels).sum().item()
        #     return correct / total

        # # Calculate and print the train, validation, and test accuracies
        # print("Calculating train acc")
        # train_accuracy = calculate_accuracy(trainloader)
        # print("Calculating val acc")
        # val_accuracy = calculate_accuracy(valloader)
        # print("Calculating test acc")
        # test_accuracy = calculate_accuracy(testloader)

        # print(f'\nTrain Accuracy: {train_accuracy}')
        # print(f'Validation Accuracy: {val_accuracy}')
        # print(f'Test Accuracy: {test_accuracy}')

        # continue



        model.reset_params() # Ensure weights are cleared from the prev. fold.
        if early_stopper:
            early_stopper.reset()

        for epoch in range(epochs):
            
            # TRAINING
            model.train()  # Sets the model to training mode.
            train_correct = 0
            total_loss = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()  # Zeros gradients for all the weights.
                # print(f"INPUTS SHAPE: {inputs.shape}")
                outputs = model(inputs)
                # print(f"OUTPUTS SHAPE: {outputs.shape}")
                try:
                    outputs = model(inputs)  # Runs the inputs through the model.
                except:
                    print(f"Error: Unable to run example through model.")
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
                          "threshold. Putting validation accuracy at 1.00")
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
                    if config['verbose']:
                        print(f"Early stopping\n")
                    print(f"Fold {fold+1}, Epoch {epoch+1}/{epochs}, "
                          f"Validation Accuracy: {val_acc*100}%")
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
        
    X_train_full = train.copy()
    if oversample:
        X_train_full = utils.oversample_underrepresented_species(
            X_train_full,
            config['species_col'],
            config['verbose']
        )
    train_dataset = Sequence_Data(
        X=X_train_full[config['seq_col']],
        y=X_train_full[config['species_col']],
        insertions=config['trainRandomInsertions'],
        deletions=config['trainRandomDeletions'],
        mutation_rate=config['trainMutationRate'],
        encoding_mode=config['encoding_mode'],
        seq_len=config['seq_target_length'])
    test_dataset = Sequence_Data(
        X=test[config['seq_col']],
        y=test[config['species_col']], 
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
    test_time = time.time() - start_time
    print(f"Test execution time: {test_time} seconds")

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
        classes=list(range(num_classes))
    )
    roc_auc = roc_auc_score(
        all_targets_one_hot,
        all_test_outputs,
        multi_class='ovr',
        average='macro'
    )

    if k_iters >= 1:
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

    # NOTE: as of 1/3, this will raise an error if you are not running this in
    # the arch search setting. This is fine, as you wouldn't want to save these
    # results anyway.
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
        'test_time': test_time,

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
        'train_mutation_rate': config['trainMutationRate'],

        # preprocessing steps. differentiates rows.
        'model_name': model.name,
        'oversample': oversample,
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

    # try:
    #     input_channels = input_channels
    # except:
    #     raise (ValueError, "Ending execution since you are probably evaluating a model,"
    #            "not performing a grid search. Skipping saving the model.")
    
    # TEMPORARILY, don't save results in the csv, just the model
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(),f'best_model_{timestamp}.pt')

    return test_acc, epoch+1

    # add the variable number of layers to the results
    for layer in range(1, len(input_channels)+1):
        results[f'layer{layer}_input_channels'] = input_channels[layer-1]
        results[f'layer{layer}_output_channels'] = output_channels[layer-1]
        results[f'layer{layer}_conv_kernel'] = conv_kernels[layer-1]
        results[f'layer{layer}_stride'] = strides[layer-1]
        results[f'layer{layer}_padding'] = paddings[layer-1]
        results[f'layer{layer}_dropout'] = dropouts[layer-1]
        results[f'layer{layer}_pool_kernel'] = pool_kernels[layer-1]

    # These results are for a single model architecture with a single set of
    # hyperparameters and a single trial.
    utils.update_results(
        results,
        compare_cols='backbone',
        model=model,
        filename=results_file_name
    )
    
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
    run_arch_search = False     # search through architectures, in addition to lr, batch size
    run_autokeras = False
    
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
        # 'train_path': './datasets/train_no_dup.csv',
        # 'test_path': './datasets/test.csv',
        'train_path': './datasets/train_oversampled_t70_noise-0_thresh-2.csv',
        'test_path': './datasets/test_t70_noise-0_thresh-2.csv',
        'sep': ';',                       # separator character in the csv file
        'species_col': 'species_cat',     # name of column containing species
        'seq_col': 'seq',                 # name of column containing sequences
        'seq_count_thresh': 2,            # ex. keep species with >1 sequences
        'test_split': 0.3,                # ex. .3 means 30% test, 70% train
        'trainRandomInsertions': [0,2],   # ex. between 0 and 2 per sequence
        'trainRandomDeletions': [0,2],    # ex. between 0 and 2 per sequence
        'trainMutationRate': 0.05,        # n*100% chance for a base to flip
        'oversample': True,              # whether or not to oversample train # OVERRIDDEN IN ARCH SEARCH
        'encoding_mode': 'probability',   # 'probability' or 'random'
        # Whether or not applying on raw unlabeled data or "clean" ref db data.
        'applying_on_raw_data': False,
        # Whether or not to augment the test set.
        'augment_test_data': True,
        'load_existing_train_test': True, # use the same train/test split as Zurich, already saved in two different csv files
        'verbose': False
    }
    if config['applying_on_raw_data']:
        config['seq_target_length'] = 150 # POSSIBLY OVERRIDDEN IN ARCH SEARCH
        config['addTagAndPrimer'] = True 
        config['addRevComplements'] = True
    elif not config['applying_on_raw_data']:
        config['seq_target_length'] = 70 # 60-70. POSSIBLY OVERRIDDEN IN ARCH SEARCH. should be EVEN
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

        # Oversampling is performed for each fold of CV, not before. This
        # prevents duplicates from existing in the train and validation sets.

        utils.print_descriptive_stats(df, cols)
        # utils.plot_species_distribution(df, species_col)

        # If you would like to verify that the train and test sets are the same
        # as in Zurich's paper, uncomment the lines below and compare files.
        # train.to_csv("/Users/Sam/OneDrive/Desktop/waggoner_train.csv")
        # test.to_csv("/Users/Sam/OneDrive/Desktop/waggoner_test.csv")
        # pause = input("PAUSE")

    elif config['load_existing_train_test']:
        cols = ['species']
        train = pd.read_csv(config['train_path'], sep=',')
        utils.print_descriptive_stats(train, cols)
        test = pd.read_csv(config['test_path'], sep=',')
        utils.print_descriptive_stats(test, cols)
        # wait = input("PAUSE")

    if run_arch_search:
        
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
        
        # The lines below set up the parameters for grid search.

        # num_trials sets the number of times each model with each set of
        # hyperparameters is run. Results are stored in a 2d list and averaged.
        num_trials = 1

        # learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 
        #                   0.01, 0.05]
        # SEARCH 1: 12/31/23
        learning_rates = [0.0005, 0.007, 0.001, 0.002]
        # SEARCH 2: 1/3/24
        learning_rates = [0.0008, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008] # expanded both ends       


        # SEARCH 1: 12/31/23
        batch_sizes = [16, 32, 64]
        # SEARCH 2: 1/3/24
        batch_sizes = [10, 16, 32, 48, 64]


        epochs = 10_000
        # Zurich called the confidence_threshold 'binzarization threshold', and
        # used 0.9 for some of their evaluations. I do not compare my results
        # to those evaluations.
        confidence_threshold = None
        early_stopper = utils.EarlyStopping(
            patience=5,
            min_pct_improvement=1, # previously 20 epochs, 0.1%
            verbose=False
        )
        k_folds = 5
        k_iters = 5 # Should be int [0 - k_folds]. Set to 0 to skip validation.

        # Grid search: Evaluates each model with a combination of
        # hyperparameters for a certain number of trials.

        # num_cnn_layers = [1, 2, 3, 4, 5]

        # num_channels = [4, 16, 32, 48, 64, 102, 128, 156, 206, 254, 355, 407]
        # conv_kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]
        # stride_lengths = [1, 2, 3]
        # padding_lengths = [1, 2, 3, 4]
        # dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # pool_kernel_sizes = [0, 2, 3, 4, 5]

        # activations = ["relu", "sigmoid", "leakyrelu"]

        # SEARCH 1: Dec. 31, 2023
        num_cnn_layers = [1, 2, 3, 4]

        num_channels = [16, 32, 64, 128, 256]
        conv_kernel_sizes = [3, 5, 7, 9, 11]
        stride_lengths = [1, 2]
        padding_lengths = [0, 1, 2]
        dropout_rates = [0, 0.2, 0.4, 0.6]
        pool_kernel_sizes = [0, 2, 3]
        

        # SEARCH 2: Jan. 3, 2024
        num_cnn_layers = [1, 2]

        num_channels = [64, 128, 196, 256, 342, 512, 612] # shifted up
        conv_kernel_sizes = [5, 6, 7, 8, 9] # middle taken
        stride_lengths = [1, 2]
        padding_lengths = [0, 1, 2]
        dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # expanded
        pool_kernel_sizes = [0, 2, 3]

        # 2/3/24
        learning_rates = [0.002]
        batch_sizes = [64]
        num_cnn_layers = [1]
        num_channels = [512]
        conv_kernel_sizes = [7]
        stride_lengths = [1]
        padding_lengths = [3]
        dropout_rates = [0.5]
        pool_kernel_sizes = [-1] # ALWAYS kernel=2, stride=2

        # 15.5 hrs = 930 minutes for 4320 architectures = ~0.215 minutes per model. actual number of models explored: 2427
        num_explorations = 10_000
        for iteration in range(num_explorations):
            print(f"\n\nIteration {iteration+1}/{num_explorations}\n\n")

            # randomly select the model archictecture
            batch_size = random.sample(batch_sizes, 1)[0]
            lr = random.sample(learning_rates, 1)[0]
            # oversample = random.choices([True, False], k=1)[0]
            oversample = True

            num_convs = random.sample(num_cnn_layers, 1)[0]
            channels = random.choices(num_channels, k=num_convs)
            conv_kernels = random.choices(conv_kernel_sizes, k=num_convs)
            strides = random.choices(stride_lengths, k=num_convs)
            paddings = random.choices(padding_lengths, k=num_convs)
            dropouts = random.choices(dropout_rates, k=num_convs)
            pool_kernels = random.choices(pool_kernel_sizes, k=num_convs)
            input_channels = [4, *channels[:-1]]
            output_channels = [*channels]

            results_file_name = 'results.csv'
            results_file_name = 'singlemodelresults.csv'

            print(f"----Model architecture: ----------")
            print(f"Input channels: {input_channels}")
            print(f"Output channels: {output_channels}")
            print(f"Convolution kernels: {conv_kernels}")
            print(f"Strides: {strides}")
            print(f"Paddings: {paddings}")
            print(f"Dropouts: {dropouts}")
            print(f"Pooling kernels: {pool_kernels}")
            print("-----Hyperparameters: -------------")
            print(f"Oversample: {oversample}")
            print(f"Batch size: {batch_size}")
            print(f"Learning rate: {lr}")
            print(f"(fixed) K-folds: {k_folds}")
            print(f"(fixed) K-iters: {k_iters}")
            print(f"------------------------------------")

            # Create a dictionary representing the current fields that
            # differentiate rows, then check if a matching row exists.

            combination = {
                    'model_name': 'VariableCNN',
                    'k_folds': k_folds,
                    'k_iters': k_iters,
                    'confidence_threshold': confidence_threshold,
                    'seq_count_threshold': config['seq_count_thresh'],
                    'seq_len': config['seq_target_length'],
                    'test_insertions': config['testRandomInsertions'],
                    'test_deletions': config['testRandomDeletions'],
                    'test_mutation_rate': config['testMutationRate'],
                    'tag_and_primer': config['addTagAndPrimer'],
                    'reverse_complements': config['addRevComplements'],
                    'encoding_mode': config['encoding_mode'],
                    'test_split': config['test_split'],
                    'load_existing_train_test': config['load_existing_train_test']
            }
            for layer in range(1, len(input_channels)+1):
                combination[f'layer{layer}_input_channels'] = input_channels[layer-1]
                combination[f'layer{layer}_output_channels'] = output_channels[layer-1]
                combination[f'layer{layer}_conv_kernel'] = conv_kernels[layer-1]
                combination[f'layer{layer}_stride'] = strides[layer-1]
                combination[f'layer{layer}_padding'] = paddings[layer-1]
                combination[f'layer{layer}_dropout'] = dropouts[layer-1]
                combination[f'layer{layer}_pool_kernel'] = pool_kernels[layer-1]

            # if you want to allow searching identical architecture/
            # hyperparameter combinations, then uncomment this function call
            if utils.check_hyperparam_originality(combination, results_file_name) != -1:
                print(f"Model architecture has already been explored. "
                    f"Exploring next random hyperparameter combination.")
                continue

            # Create the model. If the model parameters are incompatible,
            # skip to the next combination.
            try:
                model = models.VariableCNN(
                    input_channels,
                    output_channels,
                    conv_kernels,
                    strides,
                    paddings,
                    dropouts,
                    pool_kernels,
                    nn.LeakyReLU, # could include this in search
                    input_length = config['seq_target_length'],
                    num_classes = num_classes
                )
            except:
                print("Model parameters are incompatible. Exploring next "
                    "random hyperparameter combination.")
                continue
            model.to('cuda')

            for trial in range(num_trials):
                print(f"\nTraining {model.name}, Trial {trial+1}")

                # To try to get the first batch to prove it is the same as Zurich
                # first_batch = next(iter(trainloader))
                # data,labels = first_batch
                # data_path = "/Users/Sam/OneDrive/Desktop/my_first_batch_data.npy"
                # labels_path = "/Users/Sam/OneDrive/Desktop/my_first_batch_labels.npy"
                # np.save(data_path, data.numpy())
                # np.save(labels_path, labels.numpy())

                evaluate(
                    model,
                    train = train,
                    test = test,
                    k_folds = k_folds,
                    k_iters = k_iters,
                    epochs = epochs,
                    oversample = oversample,
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=lr,
                        betas=(0.9, 0.999),
                        eps=1e-07,
                        weight_decay=0.01, # was 0, go smaller (to 0.0005) if performance is bad
                        amsgrad=False),
                    loss_function = nn.CrossEntropyLoss(),
                    early_stopper = early_stopper,
                    batch_size = batch_size,
                    confidence_threshold = confidence_threshold,
                    config = config,
                    track_fold_epochs = False,
                    track_test_epochs = False,
                    num_classes = num_classes,
                    results_file_name=results_file_name
                )
                
                print(f"Total search runtime: {round((time.time() - start_time)/60,1)} minutes")

    if run_my_model:

        # cnn1 = models.CNN1(num_classes=num_classes)
        # smallcnn2 = models.SmallCNN2(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # linear1 = models.Linear1(
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # linear2 = models.Linear2(
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # zurich = models.Zurich(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn1_1 = models.SmallCNN1_1(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn1_2 = models.SmallCNN1_2(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn2_1 = models.SmallCNN2_1(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn2_2 = models.SmallCNN2_2(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn2_3 = models.SmallCNN2_3(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn2_4 = models.SmallCNN2_4(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn2_6 = models.SmallCNN2_6(
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn3 = models.SmallCNN3(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )
        # smallcnn3_1 = models.SmallCNN3_1(
        #     stride=1,
        #     in_width=config['seq_target_length'],
        #     num_classes=num_classes
        # )

        # Code for loading in all of the weights of a model:
        # best_12_31 = models.Best_12_31()
        # model_path = "saved_models/best_model_20240101_035108.pt"

        # large_best = models.Large_Best()
        # model_path = "best_model_20240103_210217.pt"

        # small_best = models.Small_Best()
        # model_path = "best_model_20240111_060457.pt"

        small_best_updated = models.Small_Best_Updated()
        
        # UNCOMMENT THIS IF YOU WANT TO LOAD THE WEIGHTS FOR A MODEL AND HARDCODE IT
        # YOU SHOULD ALSO UNCOMMENT THE PORTION IN EVALUATE()
        # model_path = "saved_models/best_model_20240111_060457.pt"
        # small_best.load_state_dict(torch.load(model_path))

        # evaluate(
        #     model=small_best,
        #     train = train,
        #     test = test,
        #     k_folds = 5,
        #     k_iters = 5,
        #     epochs = 18,
        #     oversample = True,
        #     optimizer = torch.optim.Adam(
        #         model.parameters(),
        #         lr=-1,
        #         betas=(-1, -1),
        #         eps=-1,
        #         weight_decay=-1,
        #         amsgrad=False),
        #     loss_function = nn.CrossEntropyLoss(),
        #     early_stopper = None,
        #     batch_size = 64,
        #     confidence_threshold = None,
        #     config = config,
        #     track_fold_epochs = False,
        #     track_test_epochs = False,
        #     num_classes = 156,
        #     results_file_name=None #'smallbestresults.csv'
        # )

        


        # small_best.load_state_dict(torch.load(model_path))



        # Code for loading in all of the weights EXCEPT for
        # the linear layer of a model, and training only the 
        # linear layer. Did this because I had to change the
        # pooling size on the last conv.
        # small_best = models.Small_Best()
        # model_path = "best_model_20240111_060457.pt"

        # Load state dict, excluding the linear layer
        # state_dict = torch.load(model_path)
        # state_dict.pop('linear_layer.weight', None)
        # state_dict.pop('linear_layer.bias', None)

        # large_best.load_state_dict(state_dict, strict=False)

        # # Initialize the linear layer
        # large_best.linear_layer = torch.nn.Linear(large_best.linear_layer.in_features, large_best.linear_layer.out_features)

        # Freeze all layers except the last one
        # for name, param in large_best.named_parameters():
        #     if name != 'linear_layer.weight' and name != 'linear_layer.bias':
        #         param.requires_grad = False



        start_time = time.time()
        
        # num_trials sets the number of times each model with each set of
        # hyperparameters is run. Results are stored in a 2d list and averaged.
        num_trials = 5
        learning_rates = [0.002] # 0.002 for 12-31 and small best, 0.005 for large best
        # learning_rates = [0.001]
        # learning_rates = [0.0005, 0.001]
        # learning_rates = [0.0005, 0.007, 0.001, 0.002]
        # learning_rates = [0.0005, 0.001, 0.003, 0.005]
        # learning_rates = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        # learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 
        #                   0.01, 0.05]

        batch_sizes = [64] # 16 for 12-31, 64 for large and small best
        # batch_sizes = [16, 32, 64]

        early_stopper = utils.EarlyStopping(
            patience=20,
            min_pct_improvement=.1, # previously 20 epochs, 0.1%
            verbose=False
        )
        k_folds = 5
        k_iters = 5 # Should be int [0 - k_folds]. Set to 0 to skip validation.

        oversample_options = [True]
        # oversample_options = [True, False]

        weight_decays = [0] # 4: 0.002, 0: 97.8, 94.2
        # I explored weight decays only for the top small_best model. found 0 to be best for 0.002 lr

        # This list holds all of the models that will be trained and evaluated.
        # models = [cnn1, zurich, smallcnn1_1, smallcnn1_2, smallcnn2, smallcnn2_1,
        #           smallcnn2_2, smallcnn2_3, smallcnn2_4, smallcnn2_6, smallcnn3,
        #           smallcnn3_1]
        models = [small_best_updated]

        for model in models:
            model.to('cuda')

        print(f"Evaluation for Personal Model(s):\n"
                f"{[f'{model.name}' for model in models]}")

        start_time = time.time()

        for model in models:
            model_test_accs = []
            for trial in range(num_trials):
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        for weight_decay in weight_decays:
                            for oversample in oversample_options:
                                print(f"\nTraining {model.name}, Trial {trial+1}")

                                # To try to get the first batch to prove it is the same as Zurich
                                # first_batch = next(iter(trainloader))
                                # data,labels = first_batch
                                # data_path = "/Users/Sam/OneDrive/Desktop/my_first_batch_data.npy"
                                # labels_path = "/Users/Sam/OneDrive/Desktop/my_first_batch_labels.npy"
                                # np.save(data_path, data.numpy())
                                # np.save(labels_path, labels.numpy())

                                test_acc, _ = evaluate(
                                    model,
                                    train = train,
                                    test = test,
                                    k_folds = k_folds,
                                    k_iters = k_iters,
                                    epochs = 18, #10_000, #14, # 18
                                    oversample = oversample,
                                    optimizer = torch.optim.Adam(
                                        model.parameters(),
                                        lr=lr,
                                        betas=(0.9, 0.999),
                                        eps=1e-07,
                                        weight_decay=weight_decay,
                                        amsgrad=False),
                                    loss_function = nn.CrossEntropyLoss(),
                                    early_stopper = early_stopper,
                                    batch_size = batch_size,
                                    confidence_threshold = None,
                                    config = config,
                                    track_fold_epochs = False,
                                    track_test_epochs = False,
                                    num_classes = num_classes,
                                    results_file_name='smallbestresults.csv'
                                )
                                
                                print(f"Total search runtime: {round((time.time() - start_time)/60,1)} minutes")
                                model_test_accs.append(test_acc)
            print(f"Over {num_trials} for current model, got average accuracy: {np.mean(model_test_accs)}, standard deviation: {np.std(model_test_accs)}") 
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

        # pause = input("PAUSE")



        start_time = time.time()

        # Initialize and train the classifier
        clf = ak.ImageClassifier(overwrite=False, max_trials=300)
        clf.fit(X_train, y_train) # optionally, specify epochs=5

        # Export as a Keras model then save it.
        model = clf.export_model() 
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        # now = datetime.now().strftime("%Y%m%d_%H%M%S")
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

# OLD CODE THAT CREATED A GRID FOR ARCHITECTURE SEARCH
"""
try:
            arch_grid = load('./datasets/full_arch_grid.joblib')
        except:
            print(f"Creating possible hyperparameters for grid search."
                  f"Expect ~30 minutes")
            num_cnn_layers = [1, 2, 3, 4]
            num_channels = [16, 32, 64, 128, 256]
            kernel_sizes = [3, 5, 7, 9, 11]
            strides = [1, 2]
            paddings = [1, 2]
            dropout_rates = [0, 0.2, 0.4, 0.6]
            # activations = ["relu", "sigmoid"]

            lists = [num_channels, kernel_sizes, strides, paddings, dropout_rates]

            arch_grid = []
            for length in num_cnn_layers:  # length is the length of the permutations
                print(f"Length: {length}")
                combinations = []
                for lst in lists:
                    print(f"List: {lst}")
                    list_permutation = list(itertools.product(lst, repeat=length))
                    combinations.append(list_permutation)
                print("Computing product of permutations of lists")
                combo = [list(combination) for combination in itertools.product(*combinations)]
                for combination in combo:
                    arch_grid.append(combination)
            print(len(arch_grid))
            dump(arch_grid, f'./datasets/full_arch_grid.joblib')

        print(f"Number of models to explore: {len(arch_grid)}")

        random.shuffle(arch_grid)
"""