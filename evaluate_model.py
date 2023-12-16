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

import datetime
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from tqdm import tqdm

from dataset import Sequence_Data
from model import My_CNN
import models
import utils
from torch.utils.data import DataLoader

random.seed(1327)

def evaluate(model, X_train, y_train, X_test, y_test, k_folds, epochs,
             optimizer, loss_function, early_stopper, batch_size,
             confidence_threshold, config):
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
            If None, it will not enforce a threshold.
        config (dict): A dictionary containing many hyperparameters. For a full
            list, refer to the correct usage in __main__().
            
    Returns:
        tuple: (acc, epoch), where acc is the accuracy on the test set 
        (a float between 0 and 1), and epoch is the number of epochs the model 
        was trained (an integer).
    """

    fold_val_accuracies = []
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1327)

    for fold, (train_indexes, val_indexes) in enumerate(
        skf.split(X_train, y_train)
    ):
        print(f"Starting fold {fold + 1} ------------------------------------")

        # Store the metrics below such that each list contains <epochs> elements
        # and each element is a list containing <k_folds> numbers.

        train_accuracies = [[] for _ in range(epochs)]
        val_accuracies = [[] for _ in range(epochs)]
        train_losses = [[] for _ in range(epochs)]
        val_losses = [[] for _ in range(epochs)]

        train_dataset = Sequence_Data(
            X=X_train.iloc[train_indexes],
            y=y_train.iloc[train_indexes],
            seq_col=['seq_col'],
            species_col=['species_col'],
            insertions=config['trainRandomInsertions'],
            deletions=config['trainRandomDeletions'],
            mutation_rate=config['trainMutationRate'],
            encoding_mode=config['encoding_mode'],
            iupac=config['iupac'],
            seq_len=config['seq_target_length'])
        val_dataset = Sequence_Data(
            X=X_train.iloc[val_indexes],
            y=y_train.iloc[val_indexes], 
            seq_col=['seq_col'],
            species_col=['species_col'],
            insertions=config['testRandomInsertions'],
            deletions=config['testRandomDeletions'],
            mutation_rate=config['testMutationRate'],
            encoding_mode=config['encoding_mode'],
            iupac=config['iupac'],
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

        np.set_printoptions(threshold=sys.maxsize)

        for epoch in tqdm(range(epochs)):
            
            # TRAINING
            model.train()  # Sets the model to training mode.
            train_correct = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()  # Zeros gradients for all the weights.
                outputs = model(inputs)  # Runs the inputs through the model.
                loss = loss_function(outputs, labels)  # Computes the loss.
                train_losses[epoch].append(loss)
                loss.backward()  # Performs backward pass, updating weights.
                optimizer.step()  # Performs optimization.
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()

            train_acc = train_correct / len(trainloader.dataset)
        
            # VALIDATION
            model.eval() # Set the model to evaluation mode.
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                # Iterate over the validation data and generate predictions.
                for i, (inputs, labels) in enumerate(valloader, 0):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    val_losses[epoch].append(loss)

                    if confidence_threshold:
                        # Get the predicted classes and their corresponding
                        # probabilities.
                        max_probs, predicted = torch.max(outputs, 1)
                        # Apply the confidence threshold.
                        confident_predictions = predicted[
                            max_probs > confidence_threshold
                        ]
                        confident_labels = labels[
                            max_probs > confidence_threshold
                        ]
                        # The total number of examples we are considering is no
                        # longer the length of the trainloader.
                        val_total += len(confident_predictions) 
                        comparison = confident_predictions == confident_labels
                        val_correct += comparison.sum().item()
                    else:
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                    
            if confidence_threshold:
                val_acc = val_correct/val_total
            else:
                val_acc = val_correct/len(valloader.dataset)
            
            if early_stopper:
                early_stopper(val_acc)
                if early_stopper.stop:
                    print(f"Early stopping\n")
                    fold_val_accuracies.append(val_acc)
                    break

            print(f"\n{train_correct} / {len(trainloader.dataset)} training "
                  f"examples correct")
            print(f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_acc*100}%"
                  f", Validation Accuracy: {val_acc*100}%\n")
            train_accuracies[epoch].append(train_acc)
            val_accuracies[epoch].append(val_acc)

        # This code will be reached if early stopper is not triggered and
        # the model has trained for its full number of epochs.
    
        fold_val_accuracies.append(val_acc)
    
    print(f"\n\nValidation finished.")
    print(f"Validation accuracies for each fold: {fold_val_accuracies}")
    print(f"Average validation accuracy over {k_folds} folds: "
          f"{sum(fold_val_accuracies) / k_folds}\n\n")
    
    # EVALUATION
            
    # First, retrain on the entire train dataset.
    train_dataset = Sequence_Data(
        X=X_train,
        y=y_train,
        seq_col=['seq_col'],
        species_col=['species_col'],
        insertions=config['trainRandomInsertions'],
        deletions=config['trainRandomDeletions'],
        mutation_rate=config['trainMutationRate'],
        encoding_mode=config['encoding_mode'],
        iupac=config['iupac'],
        seq_len=config['seq_target_length'])
    test_dataset = Sequence_Data(
        X=X_test,
        y=y_test, 
        seq_col=['seq_col'],
        species_col=['species_col'],
        insertions=config['testRandomInsertions'],
        deletions=config['testRandomDeletions'],
        mutation_rate=config['testMutationRate'],
        encoding_mode=config['encoding_mode'],
        iupac=config['iupac'],
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

    np.set_printoptions(threshold=sys.maxsize)

    for epoch in tqdm(range(epochs)):
        
        # TRAINING
        model.train()  # Sets the model to training mode.
        train_correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()  # Zeros gradients for all the weights.
            outputs = model(inputs)  # Runs the inputs through the model.
            loss = loss_function(outputs, labels)  # Computes the loss.
            train_losses[epoch].append(loss)
            loss.backward()  # Performs backward pass, updating weights.
            optimizer.step()  # Performs optimization.
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / len(trainloader.dataset)

        if early_stopper:
                early_stopper(train_acc)
                if early_stopper.stop:
                    print(f"Early stopping\n")
                    break

    model.eval() # Set the model to evaluation mode.
    all_test_targets = []
    all_test_predictions = []
    all_test_outputs = []

    # TESTING
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

    # Print accuracy
    correct_test_predictions = (t == p for t, p in zip(
        all_test_targets, all_test_predictions))
    num_correct_test_predictions = sum(correct_test_predictions)
    test_acc = num_correct_test_predictions / len(all_test_targets)
    print('--------------------------------')
    print(f'Model Test Accuracy: {test_acc*100} %')

    # Print precision, recall, and F-1 score
    print("Classification report for test data:")
    print(classification_report(
        all_test_targets,
        all_test_predictions,
        zero_division=0
    ))

    # Print ROC AUC for multi-class, assuming 156 classes
    all_targets_one_hot = label_binarize(
        all_test_targets,
        classes=list(range(156))
    )
    roc_auc = roc_auc_score(
        all_targets_one_hot,
        all_test_outputs,
        multi_class='ovr',
        average='weighted'
    )
    print(f'Weighted test ROC AUC score: {roc_auc}')

    # Graph results.
    # Turn the lists of tensors into lists of lists on the CPU.
    train_accuracies = utils.make_compatible_for_plotting(train_accuracies)
    train_losses = utils.make_compatible_for_plotting(train_losses)
    val_accuracies = utils.make_compatible_for_plotting(val_accuracies)
    val_losses = utils.make_compatible_for_plotting(val_losses)
    utils.graph_roc_curves(
        all_targets_one_hot,
        all_test_outputs,
        num_classes=156
    )
    # utils.graph_metric_over_time(train_accuracies, "Training", "Accuracy")
    # utils.graph_metric_over_time(train_losses, "Training", "Loss")
    # utils.graph_metric_over_time(val_accuracies, "Testing", "Accuracy")
    # utils.graph_metric_over_time(val_losses, "Testing", "Loss")
    utils.graph_train_vs_test(train_accuracies, val_accuracies, "Accuracy")
    utils.graph_train_vs_test(train_losses, val_losses, "Loss")

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

    # If set, this will skip the preprocessing and read in an existing train
    # and test csv (that are presumably already processed).
    load_existing_train_test = True
    use_my_model = True
    use_autokeras = False
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

    if not load_existing_train_test:
        df = pd.read_csv(config['data_path'], sep=config['sep'])
        print(f"Original df shape: {df.shape}")
        utils.print_descriptive_stats(df, cols)
        # utils.plot_species_distribution(df, species_col)
        le = LabelEncoder()
        df[config['species_col']] = le.fit_transform(df[config['species_col']])
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

    elif load_existing_train_test:
        train = pd.read_csv(config['train_path'], sep=',')
        test = pd.read_csv(config['test_path'], sep=',')

    if use_my_model:
        
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
        models = [smallcnn2]

        for model in models:
            model.to('cuda')

        print(f"Evaluation for Personal Model(s):\n"
              f"{[f'{model.name}' for model in models]}")
        
        # The lines below set up the parameters for grid search.

        # num_trials sets the number of times each model with each set of
        # hyperparameters is run. Results are stored in a 2d list and averaged.
        num_trials = 1
        learning_rates = [0.001]
        # learning_rates = [0.0005, 0.001]
        # learning_rates = [0.0005, 0.001, 0.003, 0.005]
        # learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 
        #                   0.01, 0.05]
        batch_size = 32
        epochs = 3 # 10_000
        # Zurich called the confidence_threshold 'binzarization threshold', and
        # used 0.9 for some of their evaluations. I do not compare my results
        # to those evaluations.
        confidence_threshold = None
        early_stopper = utils.EarlyStopping(
            patience=20,
            min_pct_improvement=0.1
        )
        k_folds = 5

        # Grid search: Evaluates each model with a combination of
        # hyperparameters for a certain number of trials.

        # This is a 2D list where each element is [model name, (learning rates,
        # epochs taken, accuracies)]
        model_results = []
                
        for model in models:
            # This list holds the sum accuracy associated with each learning rate
            #   divide by num_trials to get the average accuracy
            accuracies = [0] * len(learning_rates)
            # records the sum number of epochs taken, whether that be the specified
            #   number <epochs>, or a smaller number because of early stopping
            #   divide by num_trials to get the average num epochs taken for that lr
            max_epochs = [0] * len(learning_rates) 
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

                    # evaluate() returns the average accuracy and number of epochs taken before early stopping
                    acc, epochs_taken = evaluate(
                        model,
                        X_train = train[config['seq_col']],
                        y_train = train[config['species_col']],
                        X_test = test[config['seq_col']],
                        y_test = test[config['species_col']],
                        k_folds = k_folds,
                        epochs= epochs,
                        optimizer= torch.optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0, amsgrad=False),
                        loss_function= nn.CrossEntropyLoss(),
                        early_stopper= early_stopper,
                        batch_size = batch_size,
                        confidence_threshold= confidence_threshold,
                        config= config)
                    
                    print(f"The model took {round((time.time() - start_time)/60,1)} minutes to run.")
                    accuracies[idx] += acc
                    max_epochs[idx] += epochs_taken
            for i in range(len(learning_rates)):
                accuracies[i] = accuracies[i]/num_trials
                max_epochs[i] = max_epochs[i]/num_trials
            model_results.append((model.name, zip(learning_rates, max_epochs, accuracies)))
        print("\n##################################################################")
        print("##################################################################")
        print(f"RESULTS")
        desc_width = 50
        value_width = 20

        print(f"{'Maximum number of epochs':<{desc_width}}{epochs:>{value_width}}")
        print(f"{'Batch size':<{desc_width}}{batch_size:>{value_width}}")
        if confidence_threshold is not None:
            print(f"{'Confidence threshold (binarization)':<{desc_width}}{confidence_threshold:>{value_width}}")
        else:
            print(f"{'Confidence threshold (binarization)':<{desc_width}}{'None':>{value_width}}")
        print(f"{'Number of trials per model':<{desc_width}}{num_trials:>{value_width}}")
        print(f"{'Early stopping patience (num epochs)':<{desc_width}}{early_stopper.patience:>{value_width}}")
        print(f"{'Early stopping minimum percent improvement':<{desc_width}}{early_stopper.min_pct_improvement:>{value_width}}")

        print(f"In the form of (LR, avg epochs taken, avg accuracy)...")
        for model_result in model_results:
            print(f"{model_result[0]}:")
            for lr_combination in model_result[1]:
                print(f"\t{lr_combination}")
        print("##################################################################")
        print("##################################################################\n")

        # Set up the file
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'model_results/results_{date_time}.txt'
        if not os.path.exists('model_results'):
            os.makedirs('model_results')

        with open(filename, 'w') as f:
            f.write("\n##################################################################\n")
            f.write("##################################################################\n")
            f.write("RESULTS\n")
            
            desc_width = 50
            value_width = 20

            f.write(f"{'Maximum number of epochs':<{desc_width}}{epochs:>{value_width}}\n")
            f.write(f"{'Batch size':<{desc_width}}{batch_size:>{value_width}}\n")
            if confidence_threshold is not None:
                f.write(f"{'Confidence threshold (binarization)':<{desc_width}}{confidence_threshold:>{value_width}}\n")
            else:
                f.write(f"{'Confidence threshold (binarization)':<{desc_width}}{'None':>{value_width}}\n")
            f.write(f"{'Number of trials per model':<{desc_width}}{num_trials:>{value_width}}\n")
            f.write(f"{'Early stopping patience (num epochs)':<{desc_width}}{early_stopper.patience:>{value_width}}\n")
            f.write(f"{'Early stopping minimum percent improvement':<{desc_width}}{early_stopper.min_pct_improvement:>{value_width}}\n")
            f.write(f"In the form of (LR, avg epochs taken, avg accuracy)...")
            for model_result in model_results:
                f.write(f"{model_result[0]}:")
                for lr_combination in model_result[1]:
                    f.write(f"\t{lr_combination}")
            f.write("##################################################################\n")
            f.write("##################################################################\n")  

    if use_autokeras:

        # Import statements are included here because 1) Printing "Using
        # TensorFlow backend" to terminal may be misleading if AutoKeras is not
        # being used, and 2) it takes up to 10 seconds to import on my machine.

        import autokeras as ak
        from tensorflow.keras.models import load_model

        df = pd.read_csv(config['train_path'], sep=',')

        # Since we are using this only for finding a model, we do not care
        # about the test accuracy, we only care about finding the model that
        # produces the best test accuracy then translating it to PyTorch and
        # evaluating it. This is why we only test on the train set, and do not
        # split the data.

        train, test = df, df

        train = utils.oversample_underrepresented_species(
            train,
            config['species_col'],
            config['verbose']
        )
        X_train, y_train = utils.encode_all_data(
            train.copy(),
            config['seq_target_length'],
            config['seq_col'],
            config['species_col'],
            config['encoding_mode'],
            True,
            "np",
            True,
            config['trainRandomInsertions'],
            config['trainRandomDeletions'],
            config['trainMutationRate']
        )
        X_test, y_test = utils.encode_all_data(
            test.copy(),
            config['seq_target_length'],
            config['seq_col'],
            config['species_col'],
            config['encoding_mode'],
            True,
            "np",
            True,
            config['testRandomInsertions'],
            config['testRandomDeletions'],
            config['testMutationRate']
        )
        
        print(f"X_train SHAPE: {X_train.shape}")
        print(f"y_train SHAPE: {y_train.shape}")

        # Try AutoKeras using its image classifier.
        # Data shape is (11269, 1, 60, 4)

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