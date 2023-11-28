import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import utils
import models
import time
import sys
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from dataset import Sequence_Data  
from model import My_CNN
# from torchinfo import summary
# from augment import RandomBPFlip
from torch.utils.data import DataLoader 
import datetime

def evaluate(model, trainloader, testloader, epochs, optimizer, loss_function,
             early_stopper, confidence_threshold):
    
    if early_stopper:
        early_stopper.reset()
    train_accuracies = []
    val_accuracies = []

    np.set_printoptions(threshold=sys.maxsize)

    # TRAINING ----------------------------------------------------------------
    print(f"\n\nTraining Status:")
    for epoch in tqdm(range(epochs)):
        model.train()  # Set the model to training mode
        train_correct = 0
        
        # TRAINING ------------------------------------------------------------
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs,labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()   # zero the gradients for all the weights
            outputs = model(inputs) # run the inputs through the model
            loss = loss_function(outputs, labels) # compute the loss
            loss.backward()         # perform backward pass, updating weights
            optimizer.step()        # perform optimization
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / len(trainloader.dataset)
    
        # Evaluation using test set -------------------------------------------
        model.eval() # Set the model to evaluation mode
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, (inputs, labels) in enumerate(testloader, 0):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)

                if confidence_threshold:
                    # Get the predicted classes and their corresponding probabilities
                    max_probs, predicted = torch.max(outputs, 1)
                    # Apply the confidence threshold
                    confident_predictions = predicted[max_probs > confidence_threshold]
                    confident_labels = labels[max_probs > confidence_threshold]
                    val_total += len(confident_predictions) # since the total number of examples we are considering is no longer the length of the trainloader
                    val_correct += (confident_predictions == confident_labels).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                
        if confidence_threshold:
            val_acc = val_correct/val_total
        else:
            val_acc = val_correct/len(testloader.dataset)
        
        if early_stopper: # if using early stopping
            early_stopper(val_acc)
            if early_stopper.stop:
                print(f"Early stopping\n")
                break

        print(f"\n{train_correct} / {len(trainloader.dataset)} training examples correct")
        print(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {train_acc*100}%, Validation Accuracy: {val_acc*100}%\n')
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    
    # TESTING -----------------------------------------------------------------
    model.eval() # Set the model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        con_matrix = None
        for i, (inputs, targets) in enumerate(testloader, 0):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            # test_losses.append(loss)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Print accuracy
    acc = correct / total
    print(f"{correct}/{total} correct")
    print('--------------------------------')
    print(f'Model Accuracy: {acc*100} %')
    # print(con_matrix)
    print('--------------------------------')
    model.reset_params()
    # utils.graph_train_loss(train_losses)
    # utils.graph_train_acc(train_accuracies) # this is helpful
    # utils.graph_train_vs_test_acc(train_acc, test_acc)
    return acc, epoch+1
    



    # X_vectorized = full_dataset.x.reshape(full_dataset.x.shape[0], -1)
    # y = full_dataset.y

    # del full_dataset

    # # Fit and test a Naieve Bayes classifier using 5-fold cross-validation
    # naive_bayes_model = GaussianNB()
    # naive_bayes_scores = cross_val_score(naive_bayes_model, X_vectorized, y, cv=5)
    # print("Naive Bayes raw scores: ", naive_bayes_scores)
    # print("Naive Bayes mean score: ", np.mean(naive_bayes_scores))

    # # Fit and test a SVM classifier using 5-fold cross-validation
    # svm_model = SVC(kernel='linear', C=1, random_state=1)
    # svm_scores = cross_val_score(svm_model, X_vectorized, y, cv=3)
    # print("SVM raw scores: ", svm_scores)
    # print("SVM mean score: ", np.mean(svm_scores))

    

if __name__ == '__main__':

    '''
    assumptions:
    - csv file contains a header row
    - the number of species'categories can be represented by 'long'
    '''

    # read in all data
    # remove unused sequences with <2 seq species
    # add tags and primers to every sequence
    # oversample underrepresented species
    # add reverse complements
    # cap/fill to get 60 bp length
    # train test split
    # create train and test Datasets
        # in the Datasets, add 
        # - random insertions per sequence
        # - random deletions per sequence (either zeroing out a vector or
        #       deleting and adding a zeroed-vector to the end of the sequence)
        # - mutation rate (random flips I think)
        # - turn into a vector

    # Note: I separate the processing of the data inside and outside of the
    #   Dataset because it is easier to add noise to the base pairs than it is
    #   to add to the encoded vectors, and noise should be added in the Dataset.
    # Note: Zurich also adds 10x'n' onto the front, then the forward and
    #   backward primer (refer to page 9-10 of the published paper)

    print(f"Loading and processing data")
    
    # IUPAC ambiguity codes
    iupac = {'a':'t', 't':'a', 'c':'g', 'g':'c',
             'r':'y', 'y':'r', 'k':'m', 'm':'k', 
             'b':'v', 'd':'h', 'h':'d', 'v':'b',
             's':'w', 'w':'s', 'n':'n', 'z':'z'}

    data_path='./datasets/v4_combined_reference_sequences.csv'
    sep = ';'                       # separator character in the csv file
    species_col = 'species_cat'     # name of the column containing species
    seq_col = 'seq'                 # name of the column containing sequences
    seq_count_thresh = 2            # keep species with >1 sequence
    test_split = 0.3                # 30% test data, 70% train data
    trainRandomInsertions = [0,2]   # between 0 and 2 per sequence
    trainRandomDeletions = [0,2]    # between 0 and 2 per sequence
    trainMutationRate = 0.05        # 5% chance for a bp to be randomly flipped
    encoding_mode = 'probability'   # 'probability' or 'random'
    applying_on_raw_data = False    # applying on raw unlabeled data or ref db data
    augment_test_data = True        # whether or not to augment the test set

    if applying_on_raw_data:
        seq_target_length = 150         # cap sequences at 150bp
        addTagAndPrimer = True          # do not add tag and primer
        addRevComplements = True        # add the reverse complement of every seq
    elif not applying_on_raw_data:
        seq_target_length = 60          # cap sequences at 60bp
        addTagAndPrimer = False         # do not add tag and primer
        addRevComplements = False       # do not add the reverse complement of every seq
    """ For the evaluations, we either added no augmentation or 2% noise and
        singular insertions and deletions, as we expected the PCR amplifcation
        and sequencing to be better than the 5% noise considered during the
        training phase."""
    if augment_test_data:
        testRandomInsertions = [1,1]    # 1 per sequence
        testRandomDeletions = [1,1]     # 1 per sequence
        testMutationRate = 0.02         # 2% chance to be randomly flipped
    elif not augment_test_data:
        testRandomInsertions = [0,0]    # 0 per sequence
        testRandomDeletions = [0,0]     # 0 per sequence
        testMutationRate = 0            # 0% chance to be randomly flipped


    use_autokeras = False
    use_my_model = True
    verbose = True


    df = pd.read_csv(data_path, sep=sep)
    print(f"Original df shape: {df.shape}")
    utils.print_descriptive_stats(df, ['species','family', 'genus', 'order'])
    df = utils.remove_species_with_too_few_sequences(df, species_col, seq_count_thresh, verbose)
    if addTagAndPrimer:
        df = utils.add_tag_and_primer(df, seq_col, iupac, 'n'*10,
                        'acaccgcccgtcactct',
                        'cttccggtacacttaccatg',
                        verbose)
    if addRevComplements:
        df = utils.add_reverse_complements(df, seq_col, iupac, verbose)
    df = utils.oversample_underrepresented_species(df, species_col, verbose)
    utils.print_descriptive_stats(df, ['species','family', 'genus', 'order'])
    le = LabelEncoder()
    df[species_col] = le.fit_transform(df[species_col])

    train, test = utils.stratified_split(df, species_col, test_split)
    # Train and test are both dataframes w/ 9 columns
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")


    if use_my_model:
        train_dataset = Sequence_Data(data=train,
                                    seq_col=seq_col,
                                    species_col=species_col,
                                    insertions=trainRandomInsertions,
                                    deletions=trainRandomDeletions,
                                    mutation_rate=trainMutationRate,
                                    encoding_mode=encoding_mode,
                                    iupac=iupac,
                                    seq_len=seq_target_length)
        test_dataset = Sequence_Data(data=test, 
                                    seq_col=seq_col,
                                    species_col=species_col,
                                    insertions=testRandomInsertions,
                                    deletions=testRandomDeletions,
                                    mutation_rate=testMutationRate,
                                    encoding_mode=encoding_mode,
                                    iupac=iupac,
                                    seq_len=seq_target_length)
        
        # Example to see how data will be fed into the model
        # torch.set_printoptions(threshold=10_000_000)
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
        # pause = input("PAUSE")



        start_time = time.time()

        cnn1 = models.CNN1(num_classes=156)
        smallcnn2 = models.SmallCNN2(stride=1, in_width=seq_target_length, num_classes=156)
        linear1 = models.Linear1(in_width=seq_target_length, num_classes=156)
        linear2 = models.Linear2(in_width=seq_target_length, num_classes=156)
        zurich = models.Zurich(stride=1, in_width=seq_target_length, num_classes=156)
        smallcnn1_1 = models.SmallCNN1_1(stride=1, in_width=seq_target_length, num_classes=156)
        smallcnn1_2 = models.SmallCNN1_2(stride=1, in_width=seq_target_length, num_classes=156)
        smallcnn2_1 = models.SmallCNN2_1(stride=1, in_width=seq_target_length, num_classes=156)
        smallcnn2_2 = models.SmallCNN2_2(stride=1, in_width=seq_target_length, num_classes=156)
        smallcnn2_3 = models.SmallCNN2_3(stride=1, in_width=seq_target_length, num_classes=156)
        smallcnn3 = models.SmallCNN3(stride=1, in_width=seq_target_length, num_classes=156)
        smallcnn3_1 = models.SmallCNN3_1(stride=1, in_width=seq_target_length, num_classes=156)
        models = [smallcnn2_3]
        # models = [smallcnn2, zurich]
        # models = [cnn1, smallcnn2, linear1, linear2, zurich, smallcnn1_1, smallcnn2_1]

        for model in models:
            model.to('cuda')
            # summary(model)

        print(f"Evaluation for Personal Model(s):\n{[f'{model.name}' for model in models]}")

        num_trials = 1
        # learning_rates = [0.001]
        learning_rates = [0.0005, 0.001]
        # learning_rates = [0.001, 0.005]
        # learning_rates = [0.0005, 0.001, 0.003, 0.005]
        # learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        batch_size = 32
        epochs = 10_000
        # confidence_threshold = 0.9 # Zurich called this 'binzarization threshold'
        confidence_threshold = None
        patience = 12
        min_pct_improvement = 0.5
        early_stopper = utils.EarlyStopping(patience=patience, min_pct_improvement=min_pct_improvement)
        # early_stopper = None

        model_results = []

        for model in models:
            # records the sum accuracy associated with each learning rate
            #   divide by num_trials to get the average accuracy
            accuracies = [0] * len(learning_rates)
            # records the sum number of epochs taken, whether that be the specified
            #   number <epochs>, or a smaller number because of early stopping
            #   divide by num_trials to get the average num epochs taken for that lr
            max_epochs = [0] * len(learning_rates) 
            for idx, lr in enumerate(learning_rates):
                for trial in range(num_trials):
                    print(f"\n\nTraining Model {model.name}, Trial {trial+1}")
                    # Define data loaders for training and testing data in this fold
                    trainloader = DataLoader(
                                    train_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True)
                    testloader = DataLoader(
                                    test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)
                    acc, epochs_taken = evaluate(
                        model,
                        trainloader,
                        testloader,
                        epochs,
                        optimizer= torch.optim.Adam(model.parameters(),lr=lr),
                        loss_function= nn.CrossEntropyLoss(),
                        early_stopper= early_stopper,
                        confidence_threshold= confidence_threshold)
                    
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
        print(f"{'Early stopping patience (num epochs)':<{desc_width}}{patience:>{value_width}}")
        print(f"{'Early stopping minimum percent improvement':<{desc_width}}{min_pct_improvement:>{value_width}}")

        print(f"In the form of (LR, avg epochs taken, avg accuracy)...")
        for model_result in model_results:
            print(f"{model_result[0]}:")
            for lr_combination in model_result[1]:
                print(f"\t{lr_combination}")
        print("##################################################################")
        print("##################################################################\n")

        # Get the current date and time
        now = datetime.datetime.now()

        # Format the date and time as a string
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Define the filename using the date and time
        filename = f'model_results/results_{date_time}.txt'

        # Create directory 'model_results' if it does not exist
        if not os.path.exists('model_results'):
            os.makedirs('model_results')

        # Open the file in write mode
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
            f.write(f"{'Early stopping patience (num epochs)':<{desc_width}}{patience:>{value_width}}\n")
            f.write(f"{'Early stopping minimum percent improvement':<{desc_width}}{min_pct_improvement:>{value_width}}\n")
            f.write(f"In the form of (LR, avg epochs taken, avg accuracy)...")
            for model_result in model_results:
                f.write(f"{model_result[0]}:")
                for lr_combination in model_result[1]:
                    f.write(f"\t{lr_combination}")
            f.write("##################################################################\n")
            f.write("##################################################################\n")





        # Best Model Combinations:
        # 94.5%     SmallCNN2, 0.0005 lr, 80 epochs, 32 batch size, NO MUTATIONS



        # NO MUTATIONS SmallCNN2       
        # (0.0005, 500.0, 0.9424790356394129)
        # (0.001, 500.0, 0.9160115303983228)
        # (0.003, 500.0, 0.8583595387840671)
        # (0.005, 500.0, 0.8078485324947589)

        # MUTATIONS SmallCNN2, step=1
        # (0.0005, 500.0, 0.7443166798836902)
        # (0.008, 18.0, 0.23175355450236967)    2 trials
        # (0.001, 500.0, 0.74649748876553)
        # (0.001, 54.5, 0.9455626715462031)     2 trials
        # (0.003, 500.0, 0.3926116838487972)
        # (0.005, 500.0, 0.4037800687285223)
        # (0.005, 45.0, 0.6967063129002744)     2 trials




        # SmallCNN2, 10 epochs, 30 batch size, NO MUTATIONS
        # 0.04      0.8%, flattened after 1 epoch
        # 0.02      16%, kind of rocky, up and down
        # 0.01      0.8%, flattened after 1 epoch
        # 0.008     0.8%, flatlined from 0 epochs
        # 0.007     0.8%, flatlined from 0 epochs
        # 0.006     72%, peaked after 7 epochs
        # 0.0055    71%, peaked after 9 epochs
        # 0.005     93%, could train longer, 75%, peaked after 6 epochs
        # 0.0045    70%, could train longer
        # 0.004     80%, peaked around 5 epochs, 78%, could be trained longer
        # 0.003     87%, could train longer
        # 0.001     89%, could train longer
        # 0.0005    87%, not too much longer

        # ??? SmallCNN2, 10 epochs, 30 batch size, NO MUTATIONS
        # 0.005     93%, could train longer, 75%, peaked after 6 epochs
        # 0.003     87%, could train longer
        # 0.001     89%, could train longer
        # 0.0005    87%, not too much longer

        # SmallCNN1, 10 epochs, 30 batch size, NO MUTATIONS
        # 0.0005    90%, could train longer, 87%, slowly increasing after 10
        # 0.001     84%, slowly increasing after 10, 87%, slowly increasing after 10
        # 0.005     83%, slowly increasing after 10, 84%, train longer!
        # 0.01      78%, slowly increasing after 10, 79%, could train longer
        # 0.05      8%,  rocky but increasing after 10, 11%, rocky, slowly increasing

        # Linear1, 10 epochs, 30 batch size, NO MUTATIONS
        # 0.0005    17%, rocky, 8%
        # 0.001     7%, rocky
        # 0.005     0.8%, flatlined after 1 epoch
        # 0.01      0.8%, flatlined after 3 epochs
        # 0.05      0.5%, rocky

        # Linear2, 10 epochs, 30 batch size, NO MUTATIONS
        # 0.0005    17%, rocky, peaked at 4
        # 0.001     11%, peaked around 2
        # 0.005     3%, very rocky
        # 0.01      2%, very rocky

        # Zurich, 80 epochs, 32 batch size, NO MUTATIONS
        # 0.0005    67%, keep training!, 74%, keep training!
        # 0.001     74%, keep training?
        # 0.005     49%, peaked around 20 epochs

        # SmallCNN1_1, 10 epochs, 32 batch size, NO MUTATIONS
        # 0.0005    83%, slowly increasing at 10
        # 0.001     81%, very slowly increasing at 10
        # 0.005     77%, very slowly increasing at 10
        # 0.01      75%, slowly increasing at 10
        # 0.05      27%, rocky and slitghly decreasing

        # SmallCNN2_1, 10 epochs, 32 batch size, NO MUTATIONS
        # 0.0005    64%, still learning at 10!
        # 0.001     69%, slowly learning at 10
        # 0.005     54%, still learning at 10!
        # 0.01      58%, very slowly learning at 10
        # 0.05      .8%, getting worse

        # 0.0005    
        # 0.001     
        # 0.005     
        # 0.01      
        # 0.05      

###############################################################################
# AUTOKERAS                                                                   #
###############################################################################
    if use_autokeras:
        import autokeras as ak
        from tensorflow.keras.models import load_model
        # auto ML finds the best architecture, then you can export the model to keras
        X_train, y_train = utils.encode_all_data(train.copy(), seq_target_length, seq_col, species_col, encoding_mode, True, "np", True, trainRandomInsertions, trainRandomDeletions, trainMutationRate, iupac)
        X_test, y_test = utils.encode_all_data(test.copy(), seq_target_length, seq_col, species_col, encoding_mode, True, "np", True, testRandomInsertions, testRandomDeletions, testMutationRate, iupac)
        print(f"X_train SHAPE: {X_train.shape}")
        print(f"y_train SHAPE: {y_train.shape}")
        # print(f"FIRST ELEMENT\n{X_train.head(1)}")
        # X_train_wo_height, y_train_wo_height = utils.encode_all_data(train.copy(), seq_target_length, seq_col, species_col, encoding_mode, False, "np")
        # X_test_wo_height, y_test_wo_height = utils.encode_all_data(test.copy(), seq_target_length, seq_col, species_col, encoding_mode, False, "np")
        # print(f"X_train_wo_height SHAPE: {X_train_wo_height.shape}")
        # print(f"y_train_wo_height SHAPE: {y_train_wo_height.shape}")
        # train_df = utils.encode_all_data(train.copy(), seq_target_length, seq_col, species_col, encoding_mode, False, "df")
        # test_df = utils.encode_all_data(test.copy(), seq_target_length, seq_col, species_col, encoding_mode, False, "df")

        # Try autokeras using its image classifier ----------------------------
        # Data shape is (11269, 1, 60, 4)

        start_time = time.time()

        # Initialize and train the classifier
        clf = ak.ImageClassifier(overwrite=False, max_trials=50)
        clf.fit(X_train, y_train) # optiona2lly specify epochs=5

        # Export as a Keras Model
        model = clf.export_model()

        # Save the model
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            model.save(f"saved_models/image_model_ak_{now}", save_format="tf")
        except Exception:
            model.save(f"saved_models/image_model_ak_{now}.h5")

        # Load the model
        loaded_model = load_model(f"saved_models/image_model_ak_{now}", custom_objects=ak.CUSTOM_OBJECTS)
        print(loaded_model.summary())

        # Make predictions
        results = loaded_model.predict(X_test)

        # Convert the predictions to the same type as your labels
        predicted_labels = np.argmax(results, axis=1)

        # Calculate the accuracy
        accuracy = np.mean(predicted_labels == y_test)
        print(f"Image test accuracy: {accuracy}")
        print(f"Image ak took {round((time.time() - start_time)/60,1)} minutes to run.")

        '''
        # Try autokeras using its structured data classifier ------------------
        # Data shape is (11269, 60, 4)
        start_time = time.time()
        clf = ak.StructuredDataClassifier(overwrite=True, max_trials=1)
        # t = train_df[seq_col].to_frame()
        # t = train_df[seq_col].apply(pd.Series)
        t = pd.DataFrame(train_df[seq_col].to_numpy())
        print(t[0].shape)
        print(t[0][0].shape)
        l = train_df[species_col]
        clf.fit(t, l)
        # clf.fit(X_train_wo_height, y_train_wo_height, epochs=5)

        # Export as a Keras Model
        model = clf.export_model()

        # Save the model
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            model.save(f"saved_models/sdc_model_ak_{now}", save_format="tf")
        except Exception:
            model.save(f"saved_models/sdc_model_ak_{now}.h5")

        # Load the model
        loaded_model = load_model(f"saved_models/sdc_model_ak_{now}", custom_objects=ak.CUSTOM_OBJECTS)
        print(loaded_model.summary())

        # Make predictions
        results = loaded_model.predict(test_df[seq_col])

        # Convert the predictions to the same type as your labels
        predicted_labels = np.argmax(results, axis=1)

        # Calculate the accuracy
        accuracy = np.mean(predicted_labels == test_df[species_col].to_numpy())
        print(f"SDC test accuracy: {accuracy}")
        print(f"SDC ak took {round((time.time() - start_time)/60,1)} minutes to run.")
        '''

###############################################################################
# END AUTOKERAS                                                               #
###############################################################################

        




    







    # num_classes = len(np.unique(train_dataset.labels))

    # for conv1_out_channels in [4, 64, 128]:
    #     model = My_CNN(conv1_out_channels=conv1_out_channels, conv1_kernel_size=13,
    #                 conv2_out_channels=conv1_out_channels*2, conv2_kernel_size=13,
    #                 conv3_out_channels=conv1_out_channels*4, conv3_kernel_size=13,
    #                 n_classes=num_classes, in_len=seq_target_length).cuda()
    #     summary(model)
    #     print("--- Evaluation for CNN (2 layers)---")
    #     print("conv1_out_channels: {}".format(conv1_out_channels))
    #     evaluate(model, train_dataset, test_dataset, epochs=epochs, batch_size=30, lr=5e-5)

    # for conv1_out_channels in [4, 64, 128]:
    #     model = My_CNN(conv1_out_channels=conv1_out_channels, conv1_kernel_size=13,
    #                 conv2_out_channels=conv1_out_channels*2, conv2_kernel_size=13,
    #                 conv3_out_channels=conv1_out_channels*4, conv3_kernel_size=13,
    #                 n_classes=num_classes, in_len=seq_target_length).cuda()
    #     print("--- Evaluation for CNN (2 layers)---")
    #     print("conv1_out_channels: {}".format(conv1_out_channels))
    #     evaluate(model, train_dataset, test_dataset, epochs=epochs, batch_size=30, lr=5e-5)

    
    # given a prediction output from the model predicted_class (a list),
    # Convert to original label
    # original_label = le.inverse_transform([predicted_class])








    # '''
    # Loading in the data; the dataloader is set up
    # to work with pytorch, so this is a little bit
    # of a hacky way to use it, but it does the trick
    # '''
    # print('Kernel sizes 13, 13, 13')
    # target_sequence_length = 60 # Jon: 250
    # num_flips = 2
    # print("Augmenting with {} random flips".format(num_flips))
    # print("Trying sequence length: {}".format(target_sequence_length))
    # train_dataset = Sequence_Data(data_path='./datasets/split_datasets/Data Prep_ novel species - train d2_ ood dataset of maine genus.csv',
    #     target_sequence_length=target_sequence_length,
    #     transform=RandomBPFlip(num_flips))
    # test_dataset = Sequence_Data(data_path='./datasets/split_datasets/Data Prep_ novel species - test d2.csv',
    #     target_sequence_length=target_sequence_length)
    # n_classes = train_dataset.current_max_label
    # for conv1_out_channels in [4, 64, 128]:
    #     model = My_CNN(conv1_out_channels=conv1_out_channels, conv1_kernel_size=13,
    #                 conv2_out_channels=conv1_out_channels*2, conv2_kernel_size=13,
    #                 conv3_out_channels=conv1_out_channels*4, conv3_kernel_size=13,
    #                 n_classes=n_classes, in_len=target_sequence_length).cuda()
    #     print("--- Evaluation for CNN (2 layers)---")
    #     print("conv1_out_channels: {}".format(conv1_out_channels))
    #     evaluate(model, train_dataset, test_dataset)


# JON's OLD DATASET CODE
# if __name__ == '__main__':
#     dataset = Sequence_Data(data_path='./datasets/v4_combined_reference_sequences_no_unique_rev.csv',
#                             sep=',',                    # Jon: ','
#                             seq_col='seq',              # Jon's assumed sequence was the second column
#                             species_col='species_cat',  # Jon's assumed species was the first column
#                             target_seq_length=60,       # Jon: 250 Sam: 60 or 150
#                             seq_count_thresh=2,         # Jon: 2
#                             transform=0,                # Jon: None (would be 0 now)
#                             mode='probability'          # Jon: (would be'random' now)     
#                             )           
#     from torch.utils.data import DataLoader 
#     train_loader = DataLoader(dataset,shuffle=True,batch_size=1)
    
