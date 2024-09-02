import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
import utils

def made_dataset_with_noise(noise):
    ########################################
    train_path = '../datasets/train_oversampled_same_as_zurich.csv'
    test_path = '../datasets/test_same_as_zurich.csv'
    seq_target_length = 70 # 200
    species_col = 'species_cat' #'Species'
    seq_col = 'seq' #'Sequence'
    encoding_mode = 'probability'

    # Qualities about the existing dataset:
    oversampled = True
    for_maine_edna = False
    seq_count_thresh = 2
    same_as_zurich = True
    #########################################

    train = pd.read_csv(train_path, sep=',')
    test = pd.read_csv(test_path, sep=',')
    if for_maine_edna:
        for_maine_edna = "_maine"
    else:
        for_maine_edna = ""
    
    if oversampled:
        oversampled = "oversampled_"
    else:
        oversampled = ""
    if same_as_zurich:
        same_as_zurich = "same_as_zurich_"
    else:
        same_as_zurich = ""
    ending = f'{same_as_zurich}{oversampled}t{seq_target_length}_noise-{noise}_thresh-{seq_count_thresh}{for_maine_edna}'

    if noise == 0:
        train = utils.encode_all_data(
            train.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            False, # add noise
            [0,0],
            [0,0],
            0,
            vectorize=False
        )
        test = utils.encode_all_data(
            test.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            False, # add noise
            [0,0],
            [0,0],
            0,
            vectorize=False
        )
    elif noise == 1:
        train = utils.encode_all_data(
            train.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            True, # add noise
            [0,2],
            [0,2],
            0.05,
            vectorize=False
        )
        test = utils.encode_all_data(
            test.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            True, # add noise
            [1,1],
            [1,1],
            0.02,
            vectorize=False
        )
    elif noise == 2:
        train = utils.encode_all_data(
            train.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            True, # add noise
            [0,4],
            [0,4],
            0.1,
            vectorize=False
        )
        test = utils.encode_all_data(
            test.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            True, # add noise
            [2,2],
            [2,2],
            0.04,
            vectorize=False
        )

    X_train = train[seq_col]
    y_train = train[species_col]
    X_test = test[seq_col]
    y_test = test[species_col]
    
    print(f"X_train SHAPE: {X_train.shape}")
    print(f"y_train SHAPE: {y_train.shape}")
    print(f"X_test SHAPE: {X_test.shape}")
    print(f"y_test SHAPE: {y_test.shape}")

    train.to_csv(f'../datasets/train_{ending}.csv', index=False)
    # since the test set is not oversampled even if the training set is,
    # remove that from the name for clarity
    test.to_csv(f'../datasets/test_{ending.replace("oversampled_","")}.csv', index=False)

    print("DataFrames with added noise saved successfully to datasets folder.")

# Creates different train/test splits with different noise values.
# If you want to compare models trained and tested on different noise values, DO NOT use
# this method, as different train/test splits will overlap. Use made_dataset_with_noise() instead.
# Note: includes hardcoded values
def create_datasets():
    oversample = True
    noise = 0 # 0, 1, or 2
    for_maine_edna = False
    seq_target_length = 70 # 200
    seq_count_thresh = 2 #8
    raw_data_path = '../datasets/v4_combined_reference_sequences.csv' # '../datasets/all_data_maine.csv'
    species_col = 'species_cat' #'Species'
    seq_col = 'seq' #'Sequence'
    encoding_mode = 'probability'
    test_size = 0.3
    split_method = "fluck" # either "fluck" for fluck's custom method, or "sklearn"

    if for_maine_edna:
        for_maine_edna = "_maine"
    else:
        for_maine_edna = ""
    
    if oversample:
        oversample = "oversampled_"
    else:
        oversample = ""
    ending = f'{oversample}t{seq_target_length}_noise-{noise}_thresh-{seq_count_thresh}{for_maine_edna}'

    all_data = pd.read_csv(raw_data_path, sep=';') # for Zurich, sep=';'
    # cols = ['species']
    # print(f"Raw data:")
    # utils.print_descriptive_stats(train, cols)
    print(f"Number of unique species in all_data: {all_data[species_col].nunique()}")
    print(f"all_data shape: {all_data.shape}")
    data = utils.remove_species_with_too_few_sequences(
            all_data,
            species_col,
            seq_count_thresh,
            True # Verbose
        )
    print(f"Number of unique species in data: {data[species_col].nunique()}")
    print(f"Data shape: {data.shape}")

    le = LabelEncoder()
    data[species_col] = le.fit_transform(data[species_col])

    if split_method == "fluck":
        train, test = utils.stratified_split(data, species_col, test_size)
    elif split_method == "sklearn":
        X_train, X_test, y_train, y_test = train_test_split(
            data[seq_col], # X
            data[species_col], # y
            test_size=test_size,
            random_state=1327,
            stratify=data[species_col]
        )
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
    if oversample:
        train = utils.oversample_underrepresented_species(
            train,
            species_col,
            True # Verbose
        )
        print(f"Oversampled train shape: {train.shape}")

    if noise == 0:
        train = utils.encode_all_data(
            train.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            False, # add noise
            [0,0],
            [0,0],
            0,
            vectorize=False
        )
        test = utils.encode_all_data(
            test.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            False, # add noise
            [0,0],
            [0,0],
            0,
            vectorize=False
        )
    elif noise == 1:
        train = utils.encode_all_data(
            train.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            True, # add noise
            [0,2],
            [0,2],
            0.05,
            vectorize=False
        )
        test = utils.encode_all_data(
            test.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            True, # add noise
            [1,1],
            [1,1],
            0.02,
            vectorize=False
        )
    elif noise == 2:
        train = utils.encode_all_data(
            train.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            True, # add noise
            [0,4],
            [0,4],
            0.1,
            vectorize=False
        )
        test = utils.encode_all_data(
            test.copy(),
            seq_target_length,
            seq_col,
            species_col,
            encoding_mode,
            False, # include extra height dimension of 1
            "df", # format
            True, # add noise
            [2,2],
            [2,2],
            0.04,
            vectorize=False
        )

    X_train = train[seq_col]
    y_train = train[species_col]
    X_test = test[seq_col]
    y_test = test[species_col]
    
    print(f"X_train SHAPE: {X_train.shape}")
    print(f"y_train SHAPE: {y_train.shape}")
    print(f"X_test SHAPE: {X_test.shape}")
    print(f"y_test SHAPE: {y_test.shape}")

    train.to_csv(f'../datasets/train_{ending}.csv', index=False)
    # since the test set is not oversampled even if the training set is,
    # remove that from the name for clarity
    test.to_csv(f'../datasets/test_{ending.replace("oversampled_","")}.csv', index=False)

    print("DataFrames saved successfully.")

# create_datasets()
for noise in [0,1,2]:
    made_dataset_with_noise(noise)