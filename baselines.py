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
import warnings
import numpy as np
import pandas as pd
import time
from joblib import dump, load
import utils
from tqdm import tqdm
import os

np.set_printoptions(threshold=sys.maxsize)
# baseline_start_time = time.time()



# Optionally, truncate to 70 bases
# Sam: just do this manually and make two new csvs
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

# COMMENTED since data already has noise added

# mutation_options = {'a':'cgt', 'c':'agt', 'g':'act', 't':'acg'}
# # Defaultdict returns 'acgt' if a key is not present in the dictionary.
# mutation_options = defaultdict(lambda: 'acgt', mutation_options)

# COMMENTED OUT mutating the train set since it is already mutated

# train[config['seq_col']] = train[config['seq_col']].map(
#     lambda x: utils.add_random_insertions(
#         x,
#         config['trainRandomInsertions']
#     )
# )
# train[config['seq_col']] = train[config['seq_col']].map(
#     lambda x: utils.apply_random_deletions(
#         x,
#         config['trainRandomDeletions']
#     )
# )
# train[config['seq_col']] = train[config['seq_col']].map(
#     lambda x: utils.apply_random_mutations(
#         x,
#         config['trainMutationRate'],
#         mutation_options
#     )
# )

# COMMENTED OUT trying to directly use the sequences instead of k-mer

# # In a separate dataset, pad or truncate the vectorized data to 60bp.
# # 'z' padding will be turned to [0,0,0,0]
# train_vectorized = train.copy()
# train_vectorized[config['seq_col']] = train_vectorized[config['seq_col']].apply(
#     lambda seq: seq.ljust(config['seq_target_length'], 'z')[:config['seq_target_length']]
# )

# # In the vectorized dataset, turn every base character into a vector.
# train_vectorized[config['seq_col']] = train_vectorized[config['seq_col']].map(
#     lambda x: utils.sequence_to_array(x, config['encoding_mode'])
# )

# COMMENTED OUT since the test set is already mutated

# test[config['seq_col']] = test[config['seq_col']].map(
#     lambda x: utils.add_random_insertions(
#         x,
#         config['testRandomInsertions']
#     )
# )
# test[config['seq_col']] = test[config['seq_col']].map(
#     lambda x: utils.apply_random_deletions(
#         x,
#         config['testRandomDeletions']
#     )
# )
# test[config['seq_col']] = test[config['seq_col']].map(
#     lambda x: utils.apply_random_mutations(
#         x,
#         config['testMutationRate'],
#         mutation_options
#     )
# )

# COMMENTED OUT trying to directly use the sequences instead of k-mer

# # In a separate dataset, pad or truncate the vectorized data to 60bp.
# # 'z' padding will be turned to [0,0,0,0]
# test_vectorized = test.copy()
# test_vectorized[config['seq_col']] = test_vectorized[config['seq_col']].apply(
#     lambda seq: seq.ljust(config['seq_target_length'], 'z')[:config['seq_target_length']]
# )

# # In the vectorized dataset, turn every base character into a vector.
# test_vectorized[config['seq_col']] = test_vectorized[config['seq_col']].map(
#     lambda x: utils.sequence_to_array(x, config['encoding_mode'])
# )





def create_feature_tables(X_train, X_test, ending, include_iupac, kmer_lengths):
    # See if the feature tables k-mer feature tables (X_train and X_test) already exist
    print("Searching for pre-created k-mer feature tables...", flush=True)
    all_exist = True
    for kmer_length in kmer_lengths:
        trainpath = f'./datasets/ft_{kmer_length}{ending}.joblib'
        testpath = f'./datasets/ft_{kmer_length}_test{ending}.npy'
        if not os.path.exists(trainpath):
            all_exist = False
        if not os.path.exists(testpath):
            all_exist = False
    
    # If they don't exist, create them
    if not all_exist:
        print(f"Creating k-mer feature tables", flush=True)
        for kmer_length in kmer_lengths:
            ft_train = utils.create_feature_table_with_np(X_train, kmer_length, include_iupac)
            ft_test = utils.create_feature_table_with_np(X_test, kmer_length, include_iupac)
            np.save(f'./datasets/ft_{kmer_length}{ending}.npy', ft_train)
            np.save(f'./datasets/ft_{kmer_length}_test{ending}.npy', ft_test)

    # Visualize output
    # if not include_iupac:
    #     print(X_train[:5])
    #     print(ft_train[:5])
    # elif include_iupac:
    #     print(X_train[:16])
    #     print(ft_train[:16])

    # Could verify shapes if you wanted to by loading them back in first.
    # Modify code to do this if you want.
    # print(f"Shapes:")
    # print(f"Ft_3: {ft_train.shape}")
    # # print(f"Ft_8: {ft_8.shape}")
    # # print(f"Ft_10: {ft_10.shape}")
    # print(f"X_train: {X_train.shape}")
    # # print(f"y_train: {y_train.shape}")
    # print(f"X_test: {X_test.shape}")
    # # print(f"y_test: {y_test.shape}")


# Create, train, and save the baseline models

def train_naive_bayes(kmer, y_train, ending):
    # Naive Bayes models follow the format: nb<k-mer length>
    print("Searching for pretrained Naive Bayes models...")
    path = f'./datasets/nb{kmer}{ending}.joblib'
    
    if not os.path.exists(path):
        print("Training Naive Bayes model.", flush=True)
        nb = MultinomialNB()
        X_train = load(f'./datasets/ft_{kmer}{ending}.npy')
        nb.fit(X_train, y_train)
        dump(nb, path)

def train_svm(kmer, y_train, ending):
    # SVM models follow the format: svm<k-mer length>
    print("Searching for pretrained SVM models...")
    path = f'./datasets/svm{kmer}{ending}.joblib'
    
    if not os.path.exists(path):
        print("Training SVM model.", flush=True)
        svm_model = svm.SVC(kernel='linear', decision_function_shape='ovo')
        X_train = load(f'./datasets/ft_{kmer}{ending}.npy')
        svm_model.fit(X_train, y_train)
        dump(svm_model, path)

def train_decision_tree(kmer, y_train, ending):
    # Decision Tree models follow the format: dt<k-mer length>
    print("Searching for pretrained Decision Tree models...")
    path = f'./datasets/dt{kmer}{ending}.joblib'
    
    if not os.path.exists(path):
        print("Training Decision Tree model.", flush=True)
        dt_model = DecisionTreeClassifier(random_state=1327)
        X_train = load(f'./datasets/ft_{kmer}{ending}.npy')
        dt_model.fit(X_train, y_train)
        dump(dt_model, path)

def train_logistic_regression(kmer, y_train, ending):
    # Multiclass Logistic Regression models follow the format: lr<k-mer length>
    print("Searching for pretrained Logistic Regression models...")
    path = f'./datasets/lr{kmer}{ending}.joblib'
    
    if not os.path.exists(path):
        print("Training Logistic Regression model.", flush=True)
        lr_model = LogisticRegression(max_iter=1000, random_state=1327, solver='lbfgs', multi_class='auto')
        X_train = load(f'./datasets/ft_{kmer}{ending}.npy')
        lr_model.fit(X_train, y_train)
        dump(lr_model, path)

def train_xgboost(kmer, y_train, ending):
    # XGBoost models follow the format: xgb<k-mer length>
    print("Searching for pretrained XGBoost models...")
    path = f'./datasets/xgb{kmer}{ending}.joblib'
    
    if not os.path.exists(path):
        print("Training XGBoost model.", flush=True)
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        X_train = load(f'./datasets/ft_{kmer}{ending}.npy')
        xgb_model.fit(X_train, y_train)
        dump(xgb_model, path)

def train_knn(kmer, y_train, ending, neighbors):
    # KNN models follow the format: knn<k-mer length>_<number nearest neighbors>
    print("Searching for pretrained KNN models...")
    all_exist = True
    for neighbor in neighbors:
        path = f'./datasets/knn{kmer}_{neighbor}{ending}.joblib'
        if not os.path.exists(path):
            all_exist = False

    if not all_exist:
        print("Training KNN models.", flush=True)
        # In order to use raw data you would have to flatten it, which
        # would make it difficult for KNN since there would be 0s in the 
        # padding.
        # print(X_train_vectorized.shape)
        # print(X_train_vectorized[0]) # a 2D array, must be 1D
        # print(y_train.shape)
        # for n in neighbors:
        #     knn = KNeighborsClassifier(n_neighbors=n)
        #     knn.fit(X_train_vectorized, y_train)
        #     dump(knn, f'./datasets/knnraw_{n}.joblib')
        for n in neighbors:
            knn = KNeighborsClassifier(n_neighbors=n)
            X_train = load(f'./datasets/ft_{kmer}{ending}.npy')
            knn.fit(X_train, y_train)
            dump(knn, f'./datasets/knn{kmer}_{n}{ending}.joblib')

def train_rf(kmer, y_train, ending, num_trees):
    # Random Forest models follow the format: rf<k-mer length>_<number trees>
    print("Searching for pretrained KNN models...")
    all_exist = True
    for trees in num_trees:
        path = f'./datasets/rf{kmer}_{trees}{ending}.joblib'
        if not os.path.exists(path):
            all_exist = False

    if not all_exist:
        print("Training Random Forest models.", flush=True)
        for n in trees:
            rf_model = RandomForestClassifier(n_estimators=n, random_state=1327)
            X_train = load(f'./datasets/ft_{kmer}{ending}.npy')
            rf_model.fit(X_train, y_train)
            dump(rf_model, f'./datasets/rf{kmer}_{n}{ending}.joblib')

def train_adaboost(kmer, y_train, ending, n_estimators, max_depths):
    # AdaBoost models follow the format: adaboost<k-mer length>_<n_estimators>
    print("Searching for pretrained AdaBoost models...")
    all_exist = True
    for n in n_estimators:
        for depth in max_depths:
            path = f'./datasets/adbt{kmer}_{n}_{depth}{ending}.joblib'
            if not os.path.exists(path):
                all_exist = False

    if not all_exist:
        print("Training AdaBoost models.", flush=True)
        for n in n_estimators:
            for depth in max_depths:
                adaboost_model = AdaBoostClassifier(
                    base_estimator=DecisionTreeClassifier(max_depth=depth),
                    n_estimators=n, 
                    random_state=1327
                )
                X_train = load(f'./datasets/ft_{kmer}{ending}.npy')
                adaboost_model.fit(X_train, y_train)
                dump(adaboost_model, f'./datasets/adbt{kmer}_{n}_{depth}{ending}.joblib')
    

def evaluate(model, name, X_test, y_test, X_train, y_train):
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

    # Predictions for test set
    test_predictions = model.predict(X_test)
    
    # Metrics for test set
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_macro_f1 = f1_score(y_test, test_predictions, average="macro", zero_division=1)
    test_bal_acc = balanced_accuracy_score(y_test, test_predictions)
    test_macro_precision = precision_score(y_test, test_predictions, average="macro", zero_division=1)
    test_macro_recall = recall_score(y_test, test_predictions, average="macro", zero_division=1)
    test_weighted_precision = precision_score(y_test, test_predictions, average="weighted", zero_division=1)
    test_weighted_recall = recall_score(y_test, test_predictions, average="weighted", zero_division=1)
    test_weighted_f1 = f1_score(y_test, test_predictions, average="weighted", zero_division=1)
    try:
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="macro")
    except:
        test_roc_auc = -1
    
    # Predictions for train set
    train_predictions = model.predict(X_train)
    
    # Metrics for train set
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_macro_f1 = f1_score(y_train, train_predictions, average="macro", zero_division=1)
    train_bal_acc = balanced_accuracy_score(y_train, train_predictions)
    train_macro_precision = precision_score(y_train, train_predictions, average="macro", zero_division=1)
    train_macro_recall = recall_score(y_train, train_predictions, average="macro", zero_division=1)
    train_weighted_precision = precision_score(y_train, train_predictions, average="weighted", zero_division=1)
    train_weighted_recall = recall_score(y_train, train_predictions, average="weighted", zero_division=1)
    train_weighted_f1 = f1_score(y_train, train_predictions, average="weighted", zero_division=1)
    try:
        train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovr", average="macro")
    except:
        train_roc_auc = -1

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

        'test_macro_f1-score': test_macro_f1,
        'test_macro_recall': test_macro_recall,
        'test_micro_accuracy': test_accuracy,
        'test_macro_precision': test_macro_precision,
        'test_weighted_precision': test_weighted_precision, 
        'test_weighted_recall': test_weighted_recall,
        'test_weighted_f1-score': test_weighted_f1,
        'test_balanced_accuracy': test_bal_acc,
        'test_macro_ovr_roc_auc_score': test_roc_auc,
    }
    
    return results


    # Predict on test data and evaluate the baseline models
    # 0.00641025641 is random 1/156

    # Each element should be a tuple:
    # (trained_model, name, X_test, y_test, X_train, y_train)
    models_to_evaluate = [
        # (knnraw_1,"knnraw_1",X_test_vectorized,y_test,X_train_vectorized,y_train)
        # ,(knnraw_3,"knnraw_3",X_test_vectorized,y_test,X_train_vectorized,y_train)
        # ,(knnraw_5,"knnraw_5",X_test_vectorized,y_test,X_train_vectorized,y_train)
        # ,(knnraw_7,"knnraw_7",X_test_vectorized,y_test,X_train_vectorized,y_train)
        # ,(knnraw_9,"knnraw_9",X_test_vectorized,y_test,X_train_vectorized,y_train)
        (knn3_1,f"knn3_1{ending}",ft_3_test,y_test,ft_3,y_train)
        ,(knn3_3,f"knn3_3{ending}",ft_3_test,y_test,ft_3,y_train)
        ,(knn3_5,f"knn3_5{ending}",ft_3_test,y_test,ft_3,y_train)
        ,(knn3_7,f"knn3_7{ending}",ft_3_test,y_test,ft_3,y_train)
        ,(knn3_9,f"knn3_9{ending}",ft_3_test,y_test,ft_3,y_train)
        ,(knn5_1,f"knn5_1{ending}",ft_5_test,y_test,ft_5,y_train)
        ,(knn5_3,f"knn5_3{ending}",ft_5_test,y_test,ft_5,y_train)
        ,(knn5_5,f"knn5_5{ending}",ft_5_test,y_test,ft_5,y_train)
        ,(knn5_7,f"knn5_7{ending}",ft_5_test,y_test,ft_5,y_train)
        ,(knn5_9,f"knn5_9{ending}",ft_5_test,y_test,ft_5,y_train)
        # ,(knn8_1,f"knn8_1{ending}",ft_8_test,y_test,ft_8,y_train)
        # ,(knn8_3,f"knn8_3{ending}",ft_8_test,y_test,ft_8,y_train)
        # ,(knn8_5,f"knn8_5{ending}",ft_8_test,y_test,ft_8,y_train)
        # ,(knn8_7,f"knn8_7{ending}",ft_8_test,y_test,ft_8,y_train)
        # ,(knn8_9,f"knn8_9{ending}",ft_8_test,y_test,ft_8,y_train)
        # ,(knn10_1,f"knn10_1{ending}",ft_10_test,y_test,ft_10,y_train)
        # ,(knn10_3,f"knn10_3{ending}",ft_10_test,y_test,ft_10,y_train)
        # ,(knn10_5,f"knn10_5{ending}",ft_10_test,y_test,ft_10,y_train)
        # ,(knn10_7,f"knn10_7{ending}",ft_10_test,y_test,ft_10,y_train)
        # ,(knn10_9,f"knn10_9{ending}",ft_10_test,y_test,ft_10,y_train)

        ,(nb_3, f"nb_3{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(nb_5, f"nb_5{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(nb_8, f"nb_8{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(nb_10, f"nb_10{ending}", ft_10_test, y_test, ft_10, y_train)

        ,(rf3_15, f"rf3_15{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(rf3_30, f"rf3_30{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(rf3_45, f"rf3_45{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(rf5_15, f"rf5_15{ending}", ft_5_test, y_test, ft_5, y_train)
        ,(rf5_30, f"rf5_30{ending}", ft_5_test, y_test, ft_5, y_train)
        ,(rf5_45, f"rf5_45{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(rf8_15, f"rf8_15{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(rf8_30, f"rf8_30{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(rf8_45, f"rf8_45{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(rf10_15, f"rf10_15{ending}", ft_10_test, y_test, ft_10, y_train)
        # ,(rf10_30, f"rf10_30{ending}", ft_10_test, y_test, ft_10, y_train)
        # ,(rf10_45, f"rf10_45{ending}", ft_10_test, y_test, ft_10, y_train)

        ,(dt_3, f"dt_3{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(dt_5, f"dt_5{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(dt_8, f"dt_8{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(dt_10, f"dt_10{ending}", ft_10_test, y_test, ft_10, y_train)

        ,(svm_3, f"svm_3{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(svm_5, f"svm_5{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(svm_8, f"svm_8{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(svm_10, f"svm_10{ending}", ft_10_test, y_test, ft_10, y_train)

        ,(lr_3, f"lr_3{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(lr_5, f"lr_5{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(lr_8, f"lr_8{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(lr_10, "lr_10", ft_10_test, y_test, ft_10, y_train)

        ,(xgb_3, f"xgb_3{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(xgb_5, f"xgb_5{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(xgb_8, f"xgb_8{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(xgb_10, "xgb_10", ft_10_test, y_test, ft_10, y_train)

        ,(abdt3_15_3, f"abdt3_15_3{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt3_30_3, f"abdt3_30_3{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt3_45_3, f"abdt3_45_3{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt5_15_3, f"abdt5_15_3{ending}", ft_5_test, y_test, ft_5, y_train)
        ,(abdt5_30_3, f"abdt5_30_3{ending}", ft_5_test, y_test, ft_5, y_train)
        ,(abdt5_45_3, f"abdt5_45_3{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(abdt8_15_3, f"abdt8_15_3{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt8_30_3, f"abdt8_30_3{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt8_45_3, f"abdt8_45_3{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt10_15_3, f"abdt10_15_3{ending}", ft_10_test, y_test, ft_10, y_train)
        # ,(abdt10_30_3, f"abdt10_30_3{ending}", ft_10_test, y_test, ft_10, y_train)
        # ,(abdt10_45_3, f"abdt10_45_3{ending}", ft_10_test, y_test, ft_10, y_train)

        ,(abdt3_15_8, f"abdt3_15_8{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt3_30_8, f"abdt3_30_8{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt3_45_8, f"abdt3_45_8{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt5_15_8, f"abdt5_15_8{ending}", ft_5_test, y_test, ft_5, y_train)
        ,(abdt5_30_8, f"abdt5_30_8{ending}", ft_5_test, y_test, ft_5, y_train)
        ,(abdt5_45_8, f"abdt5_45_8{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(abdt8_15_8, f"abdt8_15_8{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt8_30_8, f"abdt8_30_8{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt8_45_8, f"abdt8_45_8{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt10_15_8, f"abdt10_15_8{ending}", ft_10_test, y_test, ft_10, y_train)
        # ,(abdt10_30_8, f"abdt10_30_8{ending}", ft_10_test, y_test, ft_10, y_train)
        # ,(abdt10_45_8, f"abdt10_45_8{ending}", ft_10_test, y_test, ft_10, y_train)

        ,(abdt3_15_18, f"abdt3_15_18{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt3_30_18, f"abdt3_30_18{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt3_45_18, f"abdt3_45_18{ending}", ft_3_test, y_test, ft_3, y_train)
        ,(abdt5_15_18, f"abdt5_15_18{ending}", ft_5_test, y_test, ft_5, y_train)
        ,(abdt5_30_18, f"abdt5_30_18{ending}", ft_5_test, y_test, ft_5, y_train)
        ,(abdt5_45_18, f"abdt5_45_18{ending}", ft_5_test, y_test, ft_5, y_train)
        # ,(abdt8_15_18, f"abdt8_15_18{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt8_30_18, f"abdt8_30_18{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt8_45_18, f"abdt8_45_18{ending}", ft_8_test, y_test, ft_8, y_train)
        # ,(abdt10_15_18, f"abdt10_15_18{ending}", ft_10_test, y_test, ft_10, y_train)
        # ,(abdt10_30_18, f"abdt10_30_18{ending}", ft_10_test, y_test, ft_10, y_train)
        # ,(abdt10_45_18, f"abdt10_45_18{ending}", ft_10_test, y_test, ft_10, y_train)
    ]
    names = [ele[1] for ele in models_to_evaluate]
    print(f"Evaluating models: {names}", flush=True)

    results_df = pd.DataFrame()

    for ele in tqdm(models_to_evaluate):
        print(f"\nRunning {ele[1]}")
        result = evaluate(*ele)
        warnings.filterwarnings('ignore', category=FutureWarning)
        results_df = results_df.append(pd.Series(result), ignore_index=True)
        warnings.filterwarnings('default', category=FutureWarning)

    results_df.to_csv('baseline_results{ending}.csv', index=False)
    # print(f"Baselines took {(baseline_start_time - time.time())/60} "
    #         f"minutes to run.")