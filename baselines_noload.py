from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn import tree, svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import sys
import random
import warnings
import numpy as np
import pandas as pd
import time
from joblib import dump, load
import utils
from tqdm import tqdm
import os
from scipy.sparse import csr_matrix

def batch_generator(X, y, batch_size):
    n_samples = X.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]

np.set_printoptions(threshold=sys.maxsize)
# baseline_start_time = time.time()

def create_feature_tables(X_train, X_test, train_ending, test_ending, include_iupac, kmer_lengths):
    # See if the feature tables k-mer feature tables (X_train and X_test) already exist
    all_exist = True
    for kmer_length in kmer_lengths:
        trainpath = f'./datasets/ft_{kmer_length}_{train_ending}.npy'
        testpath = f'./datasets/ft_{kmer_length}_{test_ending}.npy'
        # print(f"\n\n\n\nTRAINPATH: \n{trainpath}\n\n\n")
        # print(f"Trainpath: {trainpath}")
        # print(f"testpath: {testpath}")
        if not os.path.exists(trainpath):
            all_exist = False
        if not os.path.exists(testpath):
            all_exist = False
    
    # If they don't exist, create them
    if not all_exist:
        print(f"Creating k-mer feature tables for k={kmer_length}", flush=True)
        for kmer_length in kmer_lengths:
            ft_train = utils.create_feature_table_with_np(X_train, kmer_length, include_iupac)
            ft_test = utils.create_feature_table_with_np(X_test, kmer_length, include_iupac)
            np.save(f'./datasets/ft_{kmer_length}_{train_ending}.npy', ft_train)
            np.save(f'./datasets/ft_{kmer_length}_{test_ending}.npy', ft_test)
    else:
        print("Loaded pre-created k-mer feature tables", flush=True)

# Create, train, and save the baseline models

def train_naive_bayes(kmer, y_train, y_test, train_ending, test_ending, ending):
    # Naive Bayes models follow the format: nb<k-mer length>
    
    print("Training Naive Bayes model.", flush=True)
    nb = MultinomialNB()
    X_train = np.load(f'./datasets/ft_{kmer}_{train_ending}.npy')
    nb.fit(X_train, y_train)
    y_train_pred = nb.predict(X_train)
    del X_train
    X_test = np.load(f'./datasets/ft_{kmer}_{test_ending}.npy')
    y_test_pred = nb.predict(X_test)
    del X_test

    return evaluate(f'nb{kmer}_{ending}', y_test, y_test_pred, y_train, y_train_pred)


def train_svm(kmer, y_train, y_test, train_ending, test_ending, ending):
    # SVM models follow the format: svm<k-mer length>
    print("Training SVM model.", flush=True)
    svm_model = svm.SVC(kernel='linear', decision_function_shape='ovo')
    X_train = np.load(f'./datasets/ft_{kmer}_{train_ending}.npy')
    svm_model.fit(X_train, y_train)
    y_train_pred = svm_model.predict(X_train)
    del X_train
    X_test = np.load(f'./datasets/ft_{kmer}_{test_ending}.npy')
    y_test_pred = svm_model.predict(X_test)
    del X_test

    return evaluate(f'svm{kmer}_{ending}', y_test, y_test_pred, y_train, y_train_pred)

def train_decision_tree(kmer, y_train, y_test, train_ending, test_ending, ending):
    # Decision Tree models follow the format: dt<k-mer length>
    print("Training Decision Tree model.", flush=True)
    dt_model = DecisionTreeClassifier(random_state=None)
    X_train = np.load(f'./datasets/ft_{kmer}_{train_ending}.npy')
    dt_model.fit(X_train, y_train)
    y_train_pred = dt_model.predict(X_train)
    del X_train
    X_test = np.load(f'./datasets/ft_{kmer}_{test_ending}.npy')
    y_test_pred = dt_model.predict(X_test)
    del X_test

    return evaluate(f'dt{kmer}_{ending}', y_test, y_test_pred, y_train, y_train_pred)


def train_xgboost(kmer, y_train, y_test, train_ending, test_ending, ending, portions):
    # XGBoost models follow the format: xgb<k-mer length>
    X_train = np.load(f'./datasets/ft_{kmer}_{train_ending}.npy')
    X_test = np.load(f'./datasets/ft_{kmer}_{test_ending}.npy')
    results_df = pd.DataFrame()
    for p in portions:
        print(f"Training XGBoost model, portion={p}.", flush=True)
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                    random_state=None, #=random.randint(1, sys.maxsize)
                                    seed=random.randint(1, sys.maxsize),
                                    colsample_bytree=p,
                                    colsample_bylevel=p,
                                    colsample_bynode=p)
        xgb_model.fit(X_train, y_train)
        y_train_pred = xgb_model.predict(X_train)
        y_test_pred = xgb_model.predict(X_test)
        res = evaluate(f'xgb-portion{p}_{kmer}_{ending}', y_test, y_test_pred, y_train, y_train_pred)
        warnings.filterwarnings('ignore', category=FutureWarning)
        results_df = results_df.append(pd.Series(res), ignore_index=True)

    return results_df

def train_knn(kmer, y_train, y_test, train_ending, test_ending, ending, neighbors):
    # KNN models follow the format: knn<k-mer length>_<number nearest neighbors>
    X_train = np.load(f'./datasets/ft_{kmer}_{train_ending}.npy')
    X_test = np.load(f'./datasets/ft_{kmer}_{test_ending}.npy')
    results_df = pd.DataFrame()
    for n in neighbors:
        print(f"Training KNN model, n={n}.", flush=True)
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        res = evaluate(f'knn{kmer}_{n}_{ending}', y_test, y_test_pred, y_train, y_train_pred)
        warnings.filterwarnings('ignore', category=FutureWarning)
        results_df = results_df.append(pd.Series(res), ignore_index=True)

    return results_df

def train_rf(kmer, y_train, y_test, train_ending, test_ending, ending, num_trees):
    # Random Forest models follow the format: rf<k-mer length>_<number trees>
    X_train = np.load(f'./datasets/ft_{kmer}_{train_ending}.npy')
    X_test = np.load(f'./datasets/ft_{kmer}_{test_ending}.npy')
    results_df = pd.DataFrame()
    for n in num_trees:
        print(f"Training Random Forest model, num trees = {n}.", flush=True)
        rf_model = RandomForestClassifier(n_estimators=n, random_state=None)
        rf_model.fit(X_train, y_train)
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        res = evaluate(f'rf{kmer}_{n}_{ending}', y_test, y_test_pred, y_train, y_train_pred)
        warnings.filterwarnings('ignore', category=FutureWarning)
        results_df = results_df.append(pd.Series(res), ignore_index=True)

    return results_df

def train_logistic_regression(kmer, y_train, y_test, train_ending, test_ending, ending, solver):
    # Multiclass Logistic Regression models follow the format: lr<k-mer length>
    X_train = np.load(f'./datasets/ft_{kmer}_{train_ending}.npy')
    X_test = np.load(f"./datasets/ft_{kmer}_{test_ending}.npy")
    results_df = pd.DataFrame()
    for s in solver:
        print(f"Training logistic regression model, solver={s}.", flush=True)
        lr_model = LogisticRegression(max_iter=1000, random_state=None, solver=s, multi_class='auto')
        lr_model.fit(X_train, y_train)
        y_train_pred = lr_model.predict(X_train)
        y_test_pred = lr_model.predict(X_test)
        res = evaluate(f'lr-{s}{kmer}_{ending}', y_test, y_test_pred, y_train, y_train_pred)
        warnings.filterwarnings('ignore', category=FutureWarning)
        results_df = results_df.append(pd.Series(res), ignore_index=True)

    return results_df


def train_adaboost(kmer, y_train, y_test, train_ending, test_ending, ending, n_estimators, max_depths):
    # AdaBoost models follow the format: adaboost<k-mer length>_<n_estimators>
    X_train = np.load(f'./datasets/ft_{kmer}_{train_ending}.npy')
    X_test = np.load(f'./datasets/ft_{kmer}_{test_ending}.npy')
    results_df = pd.DataFrame()
    for n in n_estimators:
        for depth in max_depths:
            print(f"Training AdaBoost model, n={n}, depth={depth}.", flush=True)
            adaboost_model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=depth),
                n_estimators=n, 
                random_state=None
            )
            adaboost_model.fit(X_train, y_train)
            y_train_pred = adaboost_model.predict(X_train)
            y_test_pred = adaboost_model.predict(X_test)
            res = evaluate(f'adbt{kmer}_{n}_{depth}_{ending}', y_test, y_test_pred, y_train, y_train_pred)
            warnings.filterwarnings('ignore', category=FutureWarning)
            results_df = results_df.append(pd.Series(res), ignore_index=True)

    return results_df

def evaluate(name, y_test, y_test_pred, y_train, y_train_pred):
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

    # Predictions for test set
    test_predictions = y_test_pred
    
    # Metrics for test set
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_macro_f1 = f1_score(y_test, test_predictions, average="macro", zero_division=1)
    test_bal_acc = balanced_accuracy_score(y_test, test_predictions)
    test_macro_precision = precision_score(y_test, test_predictions, average="macro", zero_division=1)
    test_macro_recall = recall_score(y_test, test_predictions, average="macro", zero_division=1)
    test_weighted_precision = precision_score(y_test, test_predictions, average="weighted", zero_division=1)
    test_weighted_recall = recall_score(y_test, test_predictions, average="weighted", zero_division=1)
    test_weighted_f1 = f1_score(y_test, test_predictions, average="weighted", zero_division=1)
    train_roc_auc = -1
    # try:
    #     test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="macro")
    # except:
    #     test_roc_auc = -1
    
    # Predictions for train set
    train_predictions = y_train_pred
    
    # Metrics for train set
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_macro_f1 = f1_score(y_train, train_predictions, average="macro", zero_division=1)
    train_bal_acc = balanced_accuracy_score(y_train, train_predictions)
    train_macro_precision = precision_score(y_train, train_predictions, average="macro", zero_division=1)
    train_macro_recall = recall_score(y_train, train_predictions, average="macro", zero_division=1)
    train_weighted_precision = precision_score(y_train, train_predictions, average="weighted", zero_division=1)
    train_weighted_recall = recall_score(y_train, train_predictions, average="weighted", zero_division=1)
    train_weighted_f1 = f1_score(y_train, train_predictions, average="weighted", zero_division=1)
    test_roc_auc = -1
    # try:
    #     train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovr", average="macro")
    # except:
    #     train_roc_auc = -1

    # Results
    results = {
        'name': name,

        'train_macro_f1-score': train_macro_f1,
        'train_macro_recall': train_macro_recall, 
        'train_micro_accuracy': train_accuracy,
        'train_macro_precision': train_macro_precision,
        'train_weighted_precision': train_weighted_precision, 
        'train_weighted_recall': train_weighted_recall,
        'train_weighted_f1-score': train_weighted_f1,
        'train_balanced_accuracy': train_bal_acc,
        'train_macro_ovr_roc_auc_score': train_roc_auc,

        'test_micro_accuracy': test_accuracy,
        'test_macro_f1-score': test_macro_f1,
        'test_macro_recall': test_macro_recall,
        'test_macro_precision': test_macro_precision,
        'test_weighted_precision': test_weighted_precision, 
        'test_weighted_recall': test_weighted_recall,
        'test_weighted_f1-score': test_weighted_f1,
        'test_balanced_accuracy': test_bal_acc,
        'test_macro_ovr_roc_auc_score': test_roc_auc
    }
    
    return results


    # Predict on test data and evaluate the baseline models
    # 0.00641025641 is random 1/156


# Note that for the baseline models, there is no need to 1) truncate or
        # pad the sequences to any specific length, or 2) convert the data to
        # 4D vectors. Sequences are kept as strings. This is with the sole
        # exception of KNN, with which I use both a feature table and the raw
        # data itself, which is truncated to 60 bp. Ambiguity codes do not need
        # to be included as features, they just need to be randomly chosen as
        # one of 'atgc'.
        # With that said, it is a more fair comparison to neural networks if 
        # they are truncated, padded, and ambiguity codes are considered.

oversampled = True
seq_target_length = 70 # an integer, or False. 70 for French Guiana, 200 for Maine
# noise = 1
seq_count_thresh = 2 # 2 for French Guiana, 8 for Maine
for_maine_edna = False
seq_col = "seq" # "seq" for French Guiana, "Sequence" for Maine
species_col = "species_cat" # "species_cat" for French Guiana, "Species" for Maine

if for_maine_edna:
    for_maine_edna = "_maine"
else:
    for_maine_edna = ""
if oversampled:
    oversampled = "oversampled_"
else:
    oversampled = ""

def log(string):
    if verbose:
        print(string)

verbose = False

train_test_noise_combos = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

for combo in train_test_noise_combos:
    train_noise = combo[0]
    test_noise = combo[1]
    for trial in range(3):
        print(f"\n\n\ntrain noise: {train_noise}, test noise: {test_noise}, trial: {trial}\n")

        ending = f'{oversampled}t{seq_target_length}_trainnoise-{train_noise}_testnoise-{test_noise}_thresh-{seq_count_thresh}{for_maine_edna}'
        train_ending = f'train_{oversampled}t{seq_target_length}_noise-{train_noise}_thresh-{seq_count_thresh}{for_maine_edna}'
        test_ending = f'test_t{seq_target_length}_noise-{test_noise}_thresh-{seq_count_thresh}{for_maine_edna}'

        train = pd.read_csv(f'./datasets/{train_ending}.csv', sep=',')
        test = pd.read_csv(f'./datasets/{test_ending.replace("oversampled_", "")}.csv', sep=',')

        include_iupac_as_features = False


        if include_iupac_as_features:
            include_iupac_as_features = 'iupac_'
        else:
            include_iupac_as_features = ''

        ending = f"{include_iupac_as_features}{ending}"

        X_train = train.loc[:,[seq_col]].values
        # X_train_vectorized = train_vectorized.loc[:,[config['seq_col']]].values
        y_train = train.loc[:,[species_col]].values
        X_test = test.loc[:,[seq_col]].values
        # X_test_vectorized = test_vectorized.loc[:,[config['seq_col']]].values
        y_test = test.loc[:,[species_col]].values

        # Remove the inner lists for each element.
        X_train = X_train.ravel()
        # X_train_vectorized = X_train_vectorized.ravel()
        X_test = X_test.ravel()
        # X_test_vectorized = X_test_vectorized.ravel()
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        log(f"X_train shape: {X_train.shape}")
        # print(f"X_train_vectorized shape: {X_train_vectorized.shape}")
        log(f"X_test shape: {X_test.shape}")
        log(f"y_train shape: {y_train.shape}")
        log(f"y_test shape: {y_test.shape}")
        log(f"First two entries in X_train:\n{X_train[:2]}")
        log(f"First two entries in X_test:\n{X_test[:2]}")
        log(f"First two entries in y_train:\n{y_train[:2]}")
        log(f"First two entries in y_test:\n{y_test[:2]}")

        # Instead of the previous approach, saves the numpy arrays and doesn't
        # use them as local variables. Each ML method loads a specific dataset.
        for k in [3,5,8]: #[3,5,8,10]:
            log(f"Trying to create feature table for k={k}")
            create_feature_tables(X_train, X_test, train_ending, test_ending, include_iupac_as_features, kmer_lengths=[k])

        warnings.filterwarnings('ignore', category=FutureWarning)
        res_path = f'baseline_results_{ending}.csv'
        if os.path.exists(res_path):
            results_df = pd.read_csv(res_path)
        else:
            # If the file does not exist, create an empty DataFrame with the desired columns
            results_df = pd.DataFrame(columns=[
                'name',
                'train_macro_f1-score',
                'train_macro_recall', 
                'train_micro_accuracy',
                'train_macro_precision',
                'train_weighted_precision', 
                'train_weighted_recall',
                'train_weighted_f1-score',
                'train_balanced_accuracy',
                'train_macro_ovr_roc_auc_score',
                'test_macro_f1-score',
                'test_macro_recall',
                'test_micro_accuracy',
                'test_macro_precision',
                'test_weighted_precision', 
                'test_weighted_recall',
                'test_weighted_f1-score',
                'test_balanced_accuracy',
                'test_macro_ovr_roc_auc_score'
            ])

        for kmer in [3,5,8]: #[3,5,8,10]: # 3, 5, 8, 10
            print(f"KMER={kmer}", flush=True)

            # res = train_naive_bayes(kmer, y_train, y_test, train_ending, test_ending, ending)
            # results_df = results_df.append(pd.Series(res), ignore_index=True)
            # print(f"Trained naive bayes", flush=True)
            # results_df.to_csv(res_path, index=False)

            # res = train_svm(kmer, y_train, y_test, train_ending, test_ending, ending)
            # results_df = results_df.append(pd.Series(res), ignore_index=True)
            # print(f"Trained svm", flush=True)
            # results_df.to_csv(res_path, index=False)

            res = train_decision_tree(kmer, y_train, y_test, train_ending, test_ending, ending)
            results_df = results_df.append(pd.Series(res), ignore_index=True)
            print(f"Trained dt", flush=True)
            results_df.to_csv(res_path, index=False)

            # res = train_logistic_regression(kmer, y_train, y_test, train_ending, test_ending, ending, solver=['lbfgs']) # ,'saga' takes too long
            # results_df = pd.concat([results_df, res], ignore_index=True)
            # print(f"Trained lr", flush=True)
            # results_df.to_csv(res_path, index=False)

            res = train_xgboost(kmer, y_train, y_test, train_ending, test_ending, ending, portions=[0.2,0.4,0.6,0.8])
            results_df = pd.concat([results_df, res], ignore_index=True)
            print(f"Trained xgb", flush=True)
            results_df.to_csv(res_path, index=False)

            # res = train_knn(kmer, y_train, y_test, train_ending, test_ending, ending, neighbors=[1,3,5,7]) # [1,3,5,7,9]
            # results_df = pd.concat([results_df, res], ignore_index=True)
            # print(f"Trained knn", flush=True)
            # results_df.to_csv(res_path, index=False)

            res = train_rf(kmer, y_train, y_test, train_ending, test_ending, ending, num_trees=[50, 100, 200])
            results_df = pd.concat([results_df, res], ignore_index=True)
            print(f"Trained rf", flush=True)
            results_df.to_csv(res_path, index=False)

            res = train_adaboost(kmer, y_train, y_test, train_ending, test_ending, ending, n_estimators=[50,100,200], max_depths=[1,4,7])
            results_df = pd.concat([results_df, res], ignore_index=True)
            print(f"Trained adbt", flush=True)
            results_df.to_csv(res_path, index=False)
    
    # for trial in range(5):
    #     for kmer in [10]: #[3,5,8,10]: # 3, 5, 8, 10
    #         print(f"KMER={kmer}", flush=True)

    #         res = train_naive_bayes(kmer, y_train, y_test, train_ending, test_ending, ending)
    #         results_df = results_df.append(pd.Series(res), ignore_index=True)
    #         print(f"Trained naive bayes", flush=True)
    #         results_df.to_csv(res_path, index=False)

    #         res = train_svm(kmer, y_train, y_test, train_ending, test_ending, ending)
    #         results_df = results_df.append(pd.Series(res), ignore_index=True)
    #         print(f"Trained svm", flush=True)
    #         results_df.to_csv(res_path, index=False)

    #         res = train_decision_tree(kmer, y_train, y_test, train_ending, test_ending, ending)
    #         results_df = results_df.append(pd.Series(res), ignore_index=True)
    #         print(f"Trained dt", flush=True)
    #         results_df.to_csv(res_path, index=False)

    #         if kmer < 10:
    #             res = train_logistic_regression(kmer, y_train, y_test, train_ending, test_ending, ending, solver=['lbfgs','saga'])
    #             results_df = results_df.append(pd.Series(res), ignore_index=True)
    #             print(f"Trained lr", flush=True)
    #             results_df.to_csv(res_path, index=False)

    #             res = train_xgboost(kmer, y_train, y_test, train_ending, test_ending, ending)
    #             results_df = results_df.append(pd.Series(res), ignore_index=True)
    #             print(f"Trained xgb", flush=True)
    #             results_df.to_csv(res_path, index=False)

    #         res = train_knn(kmer, y_train, y_test, train_ending, test_ending, ending, neighbors=[7])
    #         # res = train_knn(kmer, y_train, y_test, ending, neighbors=[1,3,5,7,9])
    #         results_df = pd.concat([results_df, res], ignore_index=True)
    #         print(f"Trained knn", flush=True)
    #         results_df.to_csv(res_path, index=False)

    #         res = train_rf(kmer, y_train, y_test, train_ending, test_ending, ending, num_trees=[15,30,45])
    #         results_df = pd.concat([results_df, res], ignore_index=True)
    #         print(f"Trained rf", flush=True)
    #         results_df.to_csv(res_path, index=False)

    #         res = train_adaboost(kmer, y_train, y_test, train_ending, test_ending, ending, n_estimators=[15,30,45], max_depths=[5,10,15])
    #         results_df = pd.concat([results_df, res], ignore_index=True)
    #         print(f"Trained adbt", flush=True)
    #         results_df.to_csv(res_path, index=False)

    print(f"Trained and evaluated all trials for trainnoise {train_noise} testnoise {test_noise}! Saved to file {res_path}")