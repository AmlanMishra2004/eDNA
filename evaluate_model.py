import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from dataset import Sequence_Data  
from model import My_CNN
from augment import RandomBPFlip

def evaluate(model, train_dataset, test_dataset):
    # The number of training epochs
    num_train_epochs = 300
    # Will hold the results of cross validation
    results = {}

    # K-fold Cross Validation model evaluation
    # Much of this script is taken from 
    # https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=200)
    testloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=200)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    print("Batch size 200")
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(num_train_epochs):
        print(f'Starting epoch {epoch+1}')
        loss = 0
        current_loss = 0
        
        # Iterate over the entire training set
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            # Compute loss
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            print('Loss after mini-batch %5d: %.6f' %
            (i + 1, current_loss))
            current_loss = 0.0
        
        # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        con_matrix = None
        for i, (inputs, targets) in enumerate(testloader, 0):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            '''if con_matrix is None:
                con_matrix = confusion_matrix(predicted.cpu(), targets.cpu())
            else:
                con_matrix += confusion_matrix(predicted.cpu(), targets.cpu())'''

        # Print accuracy
        print('Accuracy: %d %%' % (100.0 * correct / total))
        print(con_matrix)
        print('--------------------------------')
        result = 100.0 * (correct / total)
        model.reset_params()

    # Print fold results
    print('--------------------------------')
    print(f'Accuracy: {result} %')

    '''
    X_vectorized = full_dataset.x.reshape(full_dataset.x.shape[0], -1)
    y = full_dataset.y

    del full_dataset

    # Fit and test a Naieve Bayes classifier using 5-fold cross-validation
    naive_bayes_model = GaussianNB()
    naive_bayes_scores = cross_val_score(naive_bayes_model, X_vectorized, y, cv=5)
    print("Naive Bayes raw scores: ", naive_bayes_scores)
    print("Naive Bayes mean score: ", np.mean(naive_bayes_scores))

    # Fit and test a SVM classifier using 5-fold cross-validation
    svm_model = SVC(kernel='linear', C=1, random_state=1)
    svm_scores = cross_val_score(svm_model, X_vectorized, y, cv=3)
    print("SVM raw scores: ", svm_scores)
    print("SVM mean score: ", np.mean(svm_scores))'''


if __name__ == '__main__':
    '''
    Loading in the data; the dataloader is set up
    to work with pytorch, so this is a little bit
    of a hacky way to use it, but it does the trick
    '''
    print('Kernel sizes 13, 13, 13')
    target_sequence_length = 250
    num_flips = 2
    print("Augmenting with {} random flips".format(num_flips))
    print("Trying sequence length: {}".format(target_sequence_length))
    train_dataset = Sequence_Data(data_path='./datasets/split_datasets/Data Prep_ novel species - train d2_ ood dataset of maine genus.csv',
        target_sequence_length=target_sequence_length,
        transform=RandomBPFlip(num_flips))
    test_dataset = Sequence_Data(data_path='./datasets/split_datasets/Data Prep_ novel species - test d2.csv',
        target_sequence_length=target_sequence_length)
    n_classes = train_dataset.current_max_label
    for conv1_out_channels in [4, 64, 128]:
        model = My_CNN(conv1_out_channels=conv1_out_channels, conv1_kernel_size=13,
                    conv2_out_channels=conv1_out_channels*2, conv2_kernel_size=13,
                    conv3_out_channels=conv1_out_channels*4, conv3_kernel_size=13,
                    n_classes=n_classes, in_len=target_sequence_length).cuda()
        print("--- Evaluation for CNN (2 layers)---")
        print("conv1_out_channels: {}".format(conv1_out_channels))
        evaluate(model, train_dataset, test_dataset)


    