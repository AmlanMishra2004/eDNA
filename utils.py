# (c) 2022 ETH Zürich, Switzerland
# (c) 2023 Sam Waggoner
# License: AGPLv3

"""This file contains functions that are used by evaluate_model.py and
    dataset.py.

These functions are included here in order to make the overall process simpler
to understand in evaluate_model.py and dataset.py. This is meant to provide
some abstraction from the details in these two files. With that being said,
this file contains functions that process data, split data, print statistics
about data, and graph data. Note that in order to compare my results to those
of ETH Zürich's paper, I used several of their functions directly. Zürich's
code comes from https://doi.org/10.1038/s41598-022-13412-w. My copied functions
include: add_reverse_complements(), oversample_underrepresented_species(),
add_tag_and_primer(), and stratified_split(). Zürich's function indel_mut() was
broken down into add_random_insertions(), apply_random_deletions(), and
apply_random_mutations(). These functions are copyright 2022 ETH Zürich,
Switzerland. All functions not included in this list are written by Sam
Waggoner.
"""

import os
import warnings

from collections import defaultdict
from datetime import datetime
import random

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import tensorflow as tf
import torch
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

random.seed(42)
np.random.seed(42)


# PREPROCESSING ---------------------------------------------------------------
def add_tag_and_primer(data, column, comp, tag, forward_primer, reverse_primer,
                       verbose):
    """For every row in the specified column of data, modifies the sequence to
    be: <tag><forward_primer><original sequence><reverse complement of
    reverse_primer><reversed tag>

    Args:
        data (pandas.DataFrame): The DataFrame containing the sequence data.
        column (str): The column name in data that contains the sequences.
        comp (dict): A dictionary mapping each nucleotide to its complement.
        tag (str): The tag to be added at the start and end of the sequence.
        forward_primer (str): The forward primer to be added after the tag.
        reverse_primer (str): The reverse primer whose reverse complement is to
            be added before the reversed tag.
        verbose (bool): If True, print the shape of the new DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame after adding the tag and primers to all
        sequences.
    """
    rev_tag = tag[::-1]  # the tag in reverse order
    # Reverse the reverse_primer, then take the complement of every base.
    rev_reverse_primer = ''.join(map(lambda y: comp[y], reverse_primer[::-1]))
    data[column] = data[column].apply(
        lambda x: tag + forward_primer + x + rev_reverse_primer + rev_tag
    )
    if verbose:
        print(f"Tag and primer added shape: {data.shape}")
    return data

def add_reverse_complements(data, seq_col, comp, verbose=False):
    """Adds reverse complements of every sequence.

    This function was taken from ETH Zürich. This function returns a dataframe
    that contains the reverse complement of every sequence, in addition to all
    of the original sequences.

    Args:
        data (pandas.DataFrame): The DataFrame containing the sequence data.
        seq_col (str): The column name in data that contains the sequences.
        comp (dict): A dictionary mapping each nucleotide to its complement.
        verbose (bool, optional): If True, print the shape of the new
            DataFrame. Defaults to False.

    Returns:
        pandas.DataFrame: The DataFrame after adding the reverse complements of
        all sequences.
    """
    data_rev = data.copy()
    data_rev[seq_col] = data_rev[seq_col].apply(
        lambda x: ''.join(map(lambda y: comp[y], x[::-1]))
    ) 
    new_data = pd.concat([data, data_rev])
    if verbose:
        print(f"Reverse complements added shape: {data.shape}")
    return new_data

def remove_species_with_too_few_sequences(df, species_col, seq_count_thresh,
                                          verbose=False):
    """
    Removes any sequences that belong to species who have fewer than
    seq_count_thresh sequences.

    Args:
        df (pandas.DataFrame): The DataFrame containing the species data.
        species_col (str): The column name in df that contains the species 
            labels.
        seq_count_thresh (int): The minimum number of sequences a species must
            have to be kept.
        verbose (bool, optional): If True, print the number of rows removed and
            the new DataFrame shape. Defaults to False.

    Returns:
        pandas.DataFrame: The DataFrame after removing the species with too few
            sequences.
    """
    orig_num_rows = df.shape[0]
    species_counts = df[species_col].value_counts()
    valid_species = species_counts[species_counts >= seq_count_thresh].index
    df = df[df[species_col].isin(valid_species)]
    if verbose:
        print(f"Removed {orig_num_rows - df.shape[0]} rows while enforcing "
            f"the {seq_count_thresh} sequence threshold per species.")
        print(f"Removed rows shape: {df.shape}")
    return df

def oversample_underrepresented_species(df, species_col, verbose=False):
    """
    Returns a DataFrame with an equal number of sequences from every species.

    This function was taken from ETH Zürich. This function makes it so that
    each species will have the same number of sequences as the maximum number
    of sequences for a single species. Species are upsampled by randomly
    duplicating a sequence.

    Args:
        df (pandas.DataFrame OR pandas.Series): The data to oversample.
        species_col (str): The string name of the column in the DataFrame that
            contains the species.
        verbose (bool, optional): If True, prints the shape of the oversampled
            DataFrame. Defaults to False.

    Returns:
        pandas.DataFrame: The oversampled DataFrame.

    Raises:
        KeyError: If species_col is not in df.
    """

    # For each unique species, make a list of all of the indexes in the
    # data that belong to this species. Find out how many sequences need to
    # be added in order to reach the maximum number of sequences for any
    # species. Then, randomly sample this number of indexes from the list of 
    # indexes for the current species, adding to a list of indexes for all of
    # the species' sequences that will be included at the end.

    if isinstance(df, pd.DataFrame):
        value_counts = df[species_col].value_counts()
        seq_counts = value_counts.to_dict()
        max_count = max(value_counts)
        new_indexes = []

        for val in seq_counts.keys():
            species_indexes = [i for i, j in enumerate(df[species_col])if j == val]
            new_indexes.extend(species_indexes)
            k = round(max_count - seq_counts[val])
            new_indexes.extend(random.choices(species_indexes, k=k))

        oversample = df.iloc[new_indexes]
        if verbose:
            print(f"Oversampled shape: {df.shape}")
        return oversample
    
    elif isinstance(df, pd.Series):
        value_counts = df.value_counts()
        seq_counts = value_counts.to_dict()
        max_count = max(value_counts)
        new_indexes = []

        for val in seq_counts.keys():
            species_indexes = [i for i, j in enumerate(df) if j == val]
            new_indexes.extend(species_indexes)
            k = round(max_count - seq_counts[val])
            new_indexes.extend(random.choices(species_indexes, k=k))

        oversample = df.iloc[new_indexes]
        if verbose:
            print(f"Oversampled shape: {df.shape}")
        return oversample

def stratified_split(data, species_col, ratio):
    """
    Splits data into train and test sets, stratified on a given column.

    This function ensures that at least one of each species is kept in both the
    train and test sets. This may result in a different ratio of train/test
    than the supplied ratio.

    Args:
        data (pandas.DataFrame): The DataFrame to split.
        species_col (str): The name of the column in DataFrame to stratify.
        ratio (float): The decimal representing the portion of the data that
            will be used for testing.

    Returns:
        tuple: Two pandas.DataFrame objects holding the train and test sets, in
            that order.

    Raises:
        KeyError: If species_col is not in data.
    """
    to_keep = []
    to_hold = []
    
    counts = data[species_col].value_counts()
    # For each species and its frequency (number of rows in data)...
    for t in counts.items():

        # Get a list of the indexes of every row for the current species.
        taxa_list = data[data[species_col]==t[0]][species_col].index.to_list()
        random.shuffle(taxa_list)

        # Split the shuffled list of indexes, keeping at least 1 sequence 
        # in the validation set. extend() is like .append(), except it adds
        # items individually.

        split = max(1, round(t[1]*ratio))
        to_hold.extend(taxa_list[:split])
        to_keep.extend(taxa_list[split:])
    to_keep = data.loc[to_keep]
    to_hold = data.loc[to_hold]
    return to_keep, to_hold

def print_descriptive_stats(data, cols):
    """Prints descriptive statistics for specified columns in a DataFrame.

    This function calculates and prints the number of sequences, the number of
    distinct values, the average number of sequences per distinct value, and
    the range in the number of sequences for each specified column in the
    given DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame to analyze.
        cols (list of str): The names of the columns in data to analyze.

    Raises:
        KeyError: If any element of cols is not a column name in data.
    """
    print('-' * 60)
    print(f"{'Number of sequences:':<50} {len(data):>10}")
    for col in cols:
        num_distinct = data[col].nunique()
        print(f"{'Number of distinct ' + col:<50} {num_distinct:>10}")

        avg_sequences = round(len(data) / num_distinct, 2)
        print(f"{'Average number of sequences per ' + col:<50}"
              f"{avg_sequences:>10.2f}")

        value_counts = data[col].value_counts()
        range_sequences = value_counts.max() - value_counts.min()
        print(f"{'Range in the number of sequences per ' + col:<50}"
              f"{range_sequences:>10}")
    print('-' * 60)

def plot_species_distribution(df, species_col):
    """Plots the distribution of species in a given DataFrame.

    This function takes a DataFrame and the name of the column containing
    species data. It then plots a bar chart showing the count of each species.

    Args:
        df (pandas.DataFrame): The DataFrame containing the species data.
        species_col (str): The string name of the column in df that contains
            the species data.

    Raises:
        KeyError: If species_col is not in df.
    """
    x = df[species_col]
    species_counts = x.value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(species_counts.index, species_counts.values, color='skyblue')
    plt.title('Species Distribution')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.show()

def sequence_to_array(sequence, mode):
    """Converts an string of DNA bases to a 4 channel numpy array.

    This function converts a string of DNA bases to a 4 channel numpy array,
    with A -> channel 0, T -> channel 1, C -> channel 2, and G -> channel 3.
    For probability mode, it assigns fractional values for ambiguity codes.
    For example, N would be encoded as [.25, .25, .25, .25] since N means that
    the base could be either A, T, C, or G. A full list of these codes can be
    found at the https://droog.gs.washington.edu/mdecode/images/iupac.html 
    or https://www.dnabaser.com/articles/IUPAC%20ambiguity%20codes.html. In
    random mode, it randomly chooses one of the possible bases and replaces the
    ambiguity code with the chosen base, instead of using decimals. For
    example, N could be encoded as [0, 1, 0, 0] or [0, 0, 0, 1] or other ways.
    If a character is not a member of the IUPAC ambiguity codes, then it will
    be encoded as [0, 0, 0, 0]. (This is how padding bases are encoded.)

    Args:
        sequence (str): A string of bases, can be lowercase or uppercase.
        mode (str): Either 'probability' or 'random', determines how the
            function deals with IUPAC ambiguity codes, as discussed above.

    Returns:
        numpy.ndarray: A 4 x str_len array where the associated bases are the
            nth entries of every vector. For example,
            sequence_to_array('atcg', 'probability') would return
            [[1 0 0 0], [0 1 0 0], [0 0 0 1], [0 0 1 0]].
    """
    if mode == 'probability':
        mapping = {
            'a':[1, 0, 0, 0],
            't':[0, 1, 0, 0],
            'u':[0, 1, 0, 0], # u = t
            'c':[0, 0, 1, 0],
            'g':[0, 0, 0, 1],
            # two options
            'y':[0, 0.5, 0.5 ,0],
            'r':[0.5, 0, 0, 0.5],
            'w':[0.5, 0.5, 0, 0],
            's':[0, 0, 0.5, 0.5],
            'k':[0, 0.5, 0, 0.5],
            'm':[0.5, 0, 0.5, 0],
            # three options
            'd':[1/3, 1/3, 0, 1/3],
            'v':[1/3, 0, 1/3, 1/3],
            'h':[1/3, 1/3, 1/3, 0],
            'b':[0, 1/3, 1/3, 1/3],
            # four options
            'x':[0.25, 0.25, 0.25, 0.25],
            'n':[0.25, 0.25, 0.25, 0.25]
        }
        mapping = defaultdict(lambda: [0.25, 0.25, 0.25, 0.25], mapping)
        # If you use all 0s, it causes issues with the concat_input_latent.
        # mapping = defaultdict(lambda: [0,0,0,0], mapping)

        vector = []
        for base in sequence.lower():
            vector.append(mapping[base])

            # Since mapping is a defaultdict, any base not present in the dict
            # will be turned into [0,0,0,0]. This may not be desired
            # functionality, since you may want to know if there are incorrect
            # bases in your dataset. If you wanted to recognize this, you would
            # make 'mapping' a regular dictionary, and uncomment the following
            # code.
            # try:
            #     vector.append(mapping[base])
            # except KeyError:
            #     if base == ' ':
            #         print(f"Warning: Unknown base '{base}' not found in "
            #           "mapping for sequence {sequence}. Using [0, 0, 0, 0].")
            #     vector.append([0, 0, 0, 0])
        return np.transpose(np.array(vector)).astype(np.float32)
    
    elif mode == 'random':
        output_array = torch.zeros(4, len(sequence))

        for i, letter in enumerate(sequence):
            letter = letter.upper()
            # Actual bases
            if letter == 'A':
                output_array[0, i] = 1
            elif letter == 'T':
                output_array[1, i] = 1
            elif letter == 'C':
                output_array[2, i] = 1
            elif letter == 'G':
                output_array[3, i] = 1
            # Uncertainty codes
            elif letter == 'M':
                # A or C
                if np.random.rand() < 0.5:
                    output_array[0, i] = 1
                else:
                    output_array[2, i] = 1
            elif letter == 'R':
                # A or G
                if np.random.rand() < 0.5:
                    output_array[0, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'W':
                # A or T
                if np.random.rand() < 0.5:
                    output_array[0, i] = 1
                else:
                    output_array[1, i] = 1
            elif letter == 'S':
                # C or G
                if np.random.rand() < 0.5:
                    output_array[2, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'Y':
                # C or T
                if np.random.rand() < 0.5:
                    output_array[1, i] = 1
                else:
                    output_array[2, i] = 1
            elif letter == 'K':
                # G or T
                if np.random.rand() < 0.5:
                    output_array[1, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'V':
                # A or C or G
                rand_num = np.random.rand()
                if rand_num < 1/3:
                    output_array[0, i] = 1
                elif 1/3 < rand_num and rand_num < 2/3:
                    output_array[2, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'H':
                # A or C or T
                rand_num = np.random.rand()
                if rand_num < 1/3:
                    output_array[0, i] = 1
                elif 1/3 < rand_num and rand_num < 2/3:
                    output_array[1, i] = 1
                else:
                    output_array[2, i] = 1
            elif letter == 'D':
                # A or G or T
                rand_num = np.random.rand()
                if rand_num < 1/3:
                    output_array[0, i] = 1
                elif 1/3 < rand_num and rand_num < 2/3:
                    output_array[1, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'B':
                # C or G or T
                rand_num = np.random.rand()
                if rand_num < 1/3:
                    output_array[1, i] = 1
                elif 1/3 < rand_num and rand_num < 2/3:
                    output_array[2, i] = 1
                else:
                    output_array[3, i] = 1
            elif letter == 'N' or letter == 'Z':
                # N indicates complete uncertainty
                # Z indicates padding, that there was no base there
                rand_num = np.random.rand()
                if rand_num < 1/4:
                    output_array[0, i] = 1
                elif 1/4 < rand_num and rand_num < 2/4:
                    output_array[1, i] = 1
                elif 2/4 < rand_num and rand_num < 3/4:
                    output_array[2, i] = 1
                else:
                    output_array[3, i] = 1
            else:
                print("ERROR: Unknown base '{}' encountered at index {} in {}"
                      .format(letter, i, sequence))

        print("SHAPE\n\n\n")
        print(output_array.shape)
        return output_array




# END PREPROCESSING -----------------------------------------------------------

def add_random_insertions(seq, insertions):
    """Takes a sequence and inserts a random number of random bases. 
    
    Taken from Zürich.

    Args:
        seq (str): The sequence into which base are to be inserted.
        insertions (list): A list of two integers specifying the range of the
        number of possible insertions. For example, if insertions = [1,1], the
        function inserts 1 random base in the sequence.

    Returns:
        str: The sequence after applying the random insertions.
    """
    # print(f"Insertions in the utils function: {insertions}")
    if type(insertions[0]) is list and len(insertions) == 1:
        insertions = insertions[0]
    insertions = random.randint(insertions[0],insertions[1])
    for i in range(insertions):
        pos = random.sample(range(len(seq) + 1), 1)[0]
        seq = seq[0:pos] + random.sample(['acgt'], 1)[0] + seq[pos:]
    return seq

def apply_random_deletions(seq, deletions):
    """Takes a sequence and deletes a random number of bases from it. 

    Taken from Zürich.

    Args:
        seq (str): The sequence from which bases are to be deleted.
        deletions (list): A list of two integers specifying the range of the
            number of possible deletions. For example, if deletions = [0,2],
            the function deletes between 0 and 2 random bases in the sequence.

    Returns:
        str: The sequence after applying the random deletions.
    """
    deletions = random.randint(deletions[0], deletions[1])
    for i in range(deletions):
        pos = random.sample(range(len(seq) + 1), 1)[0]
        seq = seq[0:pos] + seq[pos+1:]
    return seq

def apply_random_mutations(seq, mutation_rate, mutation_options):
    """Applies random mutations to a given sequence.

    Each base in the sequence has a mutation_rate chance (as a decimal) of 
    switching to some other base, as defined in mutation_options. From Zürich.

    Args:
        seq (str): The original sequence to which mutations will be applied.
        mutation_rate (float): The decimal chance of each base mutating. This
            should be in the range [0,1].
        mutation_options (dict): A dictionary where keys are the original bases
            and values are lists of possible mutations.

    Returns:
        str: The sequence after applying the random mutations.
    """
    # Generate a list of random nums [0-1], the same length of the sequence.
    mutations = np.random.random(len(seq))
    for idx,char in enumerate(seq):
        if mutations[idx] < mutation_rate:
            new_char = random.sample(mutation_options[char], 1)[0]
            seq = seq[0:idx] + new_char + seq[idx+1:]
    return seq


def encode_all_data(df, seq_len, seq_col, species_col, encoding_mode,
                    include_height, format, mutate, insertions, deletions,
                    mutation_rate):
    """Turns a given dataframe into torch or tf tensors after augmenting.

    This function is an alternative to online augmentation--it creates a single
    augmented dataset that can be used for training and testing, instead of
    adding different random augmentations each time data is fetched. This
    function was created primarily so that we can use AutoKeras, since AK does
    not permit for online data augmentation.
    
    Args:
        df (pandas.DataFrame): A dataframe containing both examples and labels.
        seq_len (int): The desired number of bases for all sequences.
        seq_col (string): The string name of the column containing sequences.
        species_col (string): The string name of the column containing labels.
        encoding_mode (string): Whether bases other than 'ATGC' will be encoded
            as decimals or randomly chosen. Either 'probability' or 'random'.
        include_height (bool): Whether or not to add an extra empty dimension
            for the height of the data.
        format (string): What datatype the function will return. Either 'df',
            'torch', 'tf', or 'np'.
        mutate (bool): Whether or not to add insertions, deletions, and random
            mutations to all of the data.
        insertions (list): List of two integers that represent how many random
            insertions will be added. Ex. [0,2], inserts between 0 and 2 random
            bases for every sequence in the dataset.
        deletions (list): List of two integers that represent how many random
            deletions will be performed. Ex. [1,1], removes 1 base for every
            sequence in the dataset
        mutation_rate (float): The chance [0,1] that each base in the sequence
            is switched to a different base.

    Returns:        
        tuple: (sequences, labels), both numpy.ndarray. For PyTorch, sequences
            is an array of shape (num_sequences, channels=4, length=60). For 
            TensorFlow, sequences is an array of shape (num_sequences,
            height=1, width=60, channels=4)
    """
    if mutate:
        mutation_options = {'a':'cgt', 'c':'agt', 'g':'act', 't':'acg'}
        # Defaultdict returns 'acgt' if a key is not present in the dictionary.
        mutation_options = defaultdict(lambda: 'acgt', mutation_options)

        df[seq_col] = df[seq_col].map(
            lambda x: add_random_insertions(
                x,
                insertions
            )
        )
        df[seq_col] = df[seq_col].map(
            lambda x: apply_random_deletions(
                x,
                deletions
            )
        )
        df[seq_col] = df[seq_col].map(
            lambda x: apply_random_mutations(
                x,
                mutation_rate,
                mutation_options
            )
        )

    # Pad or truncate to 60bp. 'z' padding will be turned to [0,0,0,0] below.
    df[seq_col] = df[seq_col].apply(
        lambda seq: seq.ljust(seq_len, 'z')[:seq_len]
    )

    # Turn every base character into a vector.
    df[seq_col] = df[seq_col].map(
        lambda x: sequence_to_array(x, encoding_mode)
    )

    sequences = np.array(df[seq_col].tolist())

    if format == "df":
        return df

    if include_height:
        if format == "torch":
            raise ValueError("Include height with torch is not configured")
        elif format == "tf":
            # Convert to (11269, 4, 60) to (11269, 1, 60, 4)
            new_sequences = np.zeros(
                (sequences.shape[0], 1, seq_len, 4)
            ).astype(np.float32)
            for i in range(sequences.shape[0]):
                new_sequences[i] = sequences[i].T
            new_sequences = tf.convert_to_tensor(new_sequences)
            labels = tf.convert_to_tensor(df[species_col].values)
            return new_sequences, labels
        elif format == "np":
            # Convert to (11269, 4, 60) to (11269, 1, 60, 4)
            new_sequences = np.zeros(
                (sequences.shape[0], 1, seq_len, 4)
            ).astype(np.float32)
            for i in range(sequences.shape[0]):
                new_sequences[i] = sequences[i].T
            labels = df[species_col].to_numpy()
            return new_sequences, labels
        else:
            raise ValueError("Framework name is incorrect. Use 'torch' or 'tf'"
                             " or 'np'.")
    elif not include_height:
        if format == "torch":
            sequences = torch.from_numpy(sequences)
            labels = torch.tensor(df[species_col].values) # float32 by default
            return sequences, labels
        elif format == "tf":
            # Convert to (11269, 4, 60) to (11269, 60, 4)
            new_sequences = np.zeros(
                (sequences.shape[0], seq_len, 4)
            ).astype(np.float32)
            for i in range(sequences.shape[0]):
                new_sequences[i] = sequences[i].T
            new_sequences = tf.convert_to_tensor(new_sequences)
            labels = tf.convert_to_tensor(df[species_col].values)
            return new_sequences, labels
        elif format == "np":
            # Convert to (11269, 4, 60) to (11269, 60, 4)
            new_sequences = np.zeros(
                (sequences.shape[0], seq_len, 4)
            ).astype(np.float32)
            for i in range(sequences.shape[0]):
                new_sequences[i] = sequences[i].T
            labels = df[species_col].to_numpy()
            return new_sequences, labels
        else:
            raise ValueError("Framework name is incorrect. Use 'torch' or 'tf'"
                             " or 'np'.")

def add_metrics_to_dict(labels, predicted, epoch, val_acc, fold_val_metrics):
    labels = labels.cpu()
    predicted = predicted.cpu()
    w_precision = precision_score(labels, predicted,
                                  average='weighted',
                                  zero_division=1)
    m_precision = precision_score(labels, predicted,
                                  average='macro',
                                  zero_division=1)
    w_recall = recall_score(labels, predicted,
                            average='weighted',
                            zero_division=1)
    m_recall = recall_score(labels, predicted,
                            average='macro',
                            zero_division=1)
    w_f1 = f1_score(labels, predicted,
                    average='weighted',
                    zero_division=1)
    m_f1 = f1_score(labels, predicted,
                    average='macro',
                    zero_division=1)
    bal_acc = balanced_accuracy_score(labels, predicted)
    fold_val_metrics['precision'].append((m_precision, w_precision))
    fold_val_metrics['recall'].append((m_recall, w_recall))
    fold_val_metrics['f1'].append((m_f1, w_f1))
    fold_val_metrics['acc'].append(val_acc)
    fold_val_metrics['balanced_acc'].append(bal_acc)
    fold_val_metrics['epochs_taken'].append(epoch)
    return fold_val_metrics

# instead of checking and only saving if it is original, prevent it from 
# training in the first place if the hyperparameters and model have already been explored
# returns the index (or -1) if a matching row exists

# returns -1 if no match, else returns the index of the match
def check_hyperparam_originality(results, compare_cols, filename='results.csv'):
    st = time.time()
    # Define the columns that distinguish unique rows
    if compare_cols == "backbone":
        unique_columns = [
            'model_name', 'k_folds', 'k_iters', 'confidence_threshold',
            'seq_count_threshold', 'seq_len', 'test_insertions',
            'test_deletions', 'test_mutation_rate', 'tag_and_primer',
            'reverse_complements', 'encoding_mode', 'test_split',
            'load_existing_train_test']
        # find the number of layers
        num_layer_keys = len([key for key in results.keys() if key.startswith('layer')])
        num_layers = num_layer_keys//7
        if (num_layer_keys % 7) != 0:
            raise ValueError("Unable to correctly determine number of layers used"
                    " in a model. Please check that there are 7 entries for each "
                    " layer.")
        for layer in range(1, num_layers+1):
            unique_columns.append(f'layer{layer}_input_channels')
            unique_columns.append(f'layer{layer}_output_channels')
            unique_columns.append(f'layer{layer}_conv_kernel')
            unique_columns.append(f'layer{layer}_stride')
            unique_columns.append(f'layer{layer}_padding')
            unique_columns.append(f'layer{layer}_dropout')
            unique_columns.append(f'layer{layer}_pool_kernel')
    elif compare_cols == "ppn":
        unique_columns = [
            'num_ptypes_per_class', 'ptype_length', 'prototype_shape',
            'ptype_activation_fn', 'latent_weight', 
            'joint_features_lr', 'joint_ptypes_lr', 'warm_ptypes_lr',
            'last_layer_optimizer_lr', 'weight_decay', 'joint_lr_step_size', 
            'cross_entropy_weight', 'cluster_weight', 'separation_weight',
            'l1_weight', 'num_warm_epochs',
            'push_epochs_gap', 'push_start', 'seq_count_thresh', 
            'trainRandomInsertions', 'trainRandomDeletions',
            'trainMutationRate', 'oversample', 'encoding_mode',
            'push_encoding_mode', 'applying_on_raw_data', 'augment_test_data',
            'load_existing_train_test', 'train_batch_size', 'test_batch_size', 
            'num_classes', 'seq_target_length', 'addTagAndPrimer', 
            'addRevComplements', 'val_portion_of_train', 
        ]
    else:
        raise ValueError("compare_cols should either be 'backbone' or 'ppn'")
    
    # Load existing results if file exists
    if os.path.isfile(filename):
        # If this raises an error "No columns to parse from file", copy in the 
        # column headers of the old results file.
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=list(results.keys()))

    def is_none_or_nan(val):
        return val is None or (isinstance(val, float) and math.isnan(val))

    def compare_none_and_nan(val1, val2):
        return is_none_or_nan(val1) and is_none_or_nan(val2)
    
    def compare_numbers(num1, num2):
        try:
            if float(num1) == float(num2):
                return True
        except:
            return False
    
    for index, row in df.iterrows():
        matches = True
        for col in unique_columns:
            # If both are empty, then they match. Move to check next column
            if compare_none_and_nan(results[col], row[col]):
                print(f"Value matches")
                continue
            elif compare_numbers(results[col], row[col]):
                print(f"Value matches")
                continue
            # If they are not equal, then the row does not match. Stop
            # comparing columns and move to check the next row.
            elif str(results[col]) != str(row[col]):
                # print("Found a column that doesn't match!")
                matches = False
                break
        if matches:
            print(f"Time taken to search combination: {time.time() - st} seconds")
            return index
    print(f"Time taken to search combination: {time.time() - st} seconds")
    return -1

# results: a dict
# compare_cols: either "backbone" or "ppn", to know which columns
#   dilineate unique results
# be careful when comparing existing cells since they contain arrays (and must use .all()) as well as empty cells which are translated as NaN, and NaN != NaN.
def update_results(results, compare_cols, model=None, filename='results.csv',
                   save_model_dir='saved_models'):
    
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    results['datetime'] = timestamp

    # Load existing results if file exists
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=list(results.keys()))

    # delete these three lines and uncomment below if you want to exclude saving duplicate rows
    warnings.filterwarnings('ignore', category=FutureWarning)
    df = df.append(pd.Series(results), ignore_index=True) 
    warnings.filterwarnings('default', category=FutureWarning)
    # idx = check_hyperparam_originality(results, compare_cols, filename)

    # # If the model and hyperparameters haven't been tried before, add a new row
    # if idx == -1:
    #     results['trials'] = 1
    #     warnings.filterwarnings('ignore', category=FutureWarning)
    #     df = df.append(pd.Series(results), ignore_index=True)
    #     warnings.filterwarnings('default', category=FutureWarning)
    #     print("Logged a new row for unique model/hyperparameter combination.")

    # # If the combination has been tried before, if the performance is better
    # # then replace it and save the model parameters.
    # else:
    #     row = df.loc[idx]
    #     if results['val_macro_f1-score'] == row['val_macro_f1-score']:
    #         keys_to_average = ['val_macro_f1-score', 'val_macro_recall',
    #                             'val_micro_accuracy', 'val_macro_precision',
    #                             'val_weighted_precision',
    #                             'val_weighted_recall', 
    #                             'val_weighted_f1-score',
    #                             'val_balanced_accuracy']
    #         existing_values = [row[key] for key in keys_to_average]
    #         new_values = [results[key] for key in keys_to_average]
    #         new_average_metric = sum(new_values) / len(new_values)
    #         existing_average_metric = sum(existing_values) / len(existing_values)
    #         if new_average_metric > existing_average_metric:
    #             print("Improved a model with the same hyperparameters.")
    #             df.loc[idx] = pd.Series(results)
    #         else:
    #             print("Matched f-1 score of a model with the same "
    #                     "hyperparameters, but avg. of metrics was lower.")
    #     # If the new val_macro_f1-score is higher, replace the old row
    #     elif results['val_macro_f1-score'] > row['val_macro_f1-score']:
    #         print("Improved a model with the same hyperparameters.")
    #         df.loc[idx] = pd.Series(results)
    #     else:
    #         print("Unsuccessfully tried to improve the same model and "
    #                 "hyperparameter combination.")
    #     df.loc[idx, 'trials'] += 1

    # =MATCH(MAX(C:C), C:C, 0) returns the index of the highest micro accuracy
    # Save the updated results
    df_sorted = df.sort_values(
        by='val_micro_accuracy',
        kind='mergesort',
        ascending=False
    )
    df_sorted.to_csv(filename, index=False)

    # If the val_macro_f1-score is the best, save the model parameters
    # if results['val_macro_f1-score'] >= df_sorted.iloc[0]['val_macro_f1-score']:
    #     if results['val_micro_accuracy'] >= df_sorted.iloc[0]['val_micro_accuracy']:
    #         print("Found a new best model!")
    #         if model:
    #             torch.save(
    #                 model.state_dict(),
    #                 os.path.join(save_model_dir, f'best_model_{timestamp}.pt')
    #             )
    # save regardless of performance, since it was >= 97%.
    
    if save_model_dir:
        # Create the directory if it does not already exist
        os.makedirs(save_model_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(save_model_dir, f'best_model_{timestamp}.pt')
        )
    print("Saved Results")

def conv1d_output_size(input_length, kernel_size, padding, stride):
    return (input_length - kernel_size + 2 * padding) // stride + 1

def make_compatible_for_plotting(data):
    """Converts given data into a format compatible for plotting with
        matplotlib.

    Matplotlib does not work with PyTorch or TensorFlow tensors. If the input
    data is one of these two types, this function makes the data into a numpy
    array, which is can then be used to graph using matplotlib.
    This function handles several types of input data: 2D lists, lists of
    tensors, numpy ndarrays, PyTorch tensors, and TensorFlow tensors. For 2D
    lists, it averages the inner lists. For lists of tensors, it converts each
    tensor to a numpy array. For PyTorch and TensorFlow tensors, it ensures
    they are on the CPU before converting them to numpy arrays.

    Args:
        data: The input data. This can be a 2D list, a list of tensors, a numpy
            ndarray, a PyTorch tensor, or a TensorFlow tensor.

    Returns:
        The input data converted to a numpy array, or in the case of a 2D list,
        a list of averages of the inner lists.

    Raises:
        TypeError: If the type of the input data is not recognized.
    """
    # If the data is a 2D list, average the result of each of the K folds for
    # each epoch.
    if isinstance(data[0], list):
        data = [sum(inner_list) / len(inner_list) if len(inner_list) != 0 else
                [] for inner_list in data]
        # Remove all empty inner lists.
        data = [inner_list for inner_list in data if inner_list]

    # If data is a list, check if it contains tensors.
    if isinstance(data, list):
        # If the first element of the list is a tensor, assume all elements are
        # tensors.
        if torch.is_tensor(data[0]):
            # Convert each tensor to a numpy array.
            return [d.cpu().detach().numpy() if d.is_cuda else
                    d.detach().numpy() for d in data]
        else:
            return np.array(data)
    
    # If data is a numpy ndarray, no conversion is needed.
    elif isinstance(data, np.ndarray):
        return data
    
    # If data is a PyTorch tensor, convert it to a numpy array.
    elif torch.is_tensor(data):
        if data.is_cuda:
            return data.cpu().detach().numpy()
        else:
            return data.detach().numpy()
    
    # If data is a TensorFlow tensor, convert it to a numpy array.
    elif isinstance(data, tf.Tensor):
        if data.device.type == 'cuda':
            data = data.cpu()
        return data.numpy()
    
    else:
        raise TypeError("Data type not recognized. Please provide a list,"
                        "numpy ndarray, PyTorch tensor, or TensorFlow tensor.")

def graph_train_vs_test(train, test, acc_or_loss):
    """Plots both training and testing metrics over time (epochs).

    Args:
        train (list): A list of training metric values over epochs.
        test (list): A list of testing metric values over epochs.
        acc_or_loss (str): The name of the metric being plotted. It should be
            either "Accuracy" or "Loss". Other values should not be used.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.title(f'Train vs Test {acc_or_loss}')
    plt.xlabel('Epochs')
    plt.ylabel(acc_or_loss)
    plt.legend()
    plt.show()

def graph_metric_over_time(l, training_or_testing, metric):
    """Plots a single given metric over time (epochs).

    Args:
        l (list): A list of metric values over epochs. For example, this could
            be the accuracy for each epoch, or the loss for each epoch.
        training_or_testing (str): A string indicating whether the data is from
            "Training" or "Testing". Other values should not be used.
        metric (str): The name of the metric being plotted. Ex. "Accuracy"
    """
    plt.figure(figsize=(10, 5))
    label = f"{training_or_testing} {metric}"
    plt.plot(range(1, len(l) + 1), l, label=label)
    plt.title(f"{label} vs Epochs")
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()

def graph_roc_curves(all_targets_one_hot, all_outputs, num_classes=156):
    """Plots Receiver Operating Characteristic (ROC) curves for each class.

    Args:
        all_targets_one_hot (np.array): A 2D numpy array where each row is the
            one-hot encoded target class.
        all_outputs (list): A list of output probabilities from the model. Each
            element is a list of probabilities corresponding to each class.
        num_classes (int, optional): The number of classes. Defaults to 156.
    """
    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_targets_one_hot[:, i],
                                np.array(all_outputs)[:, i]
                                )
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f) for class %i'
                  % (roc_auc, i))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for each class')
    plt.show()

class EarlyStopping:
    """A class used for adding early stopping to a neural network.
    
    The user should log the validation accuracy after each epoch, like 
    early_stopper_instance(validation_acc). After doing this, the object will
    indicate when training should be stopped by setting self.stop to True if
    the accuracy hasn't improved by <min_pct_improvement> percent after
    <patience> number of epochs. 

    Attributes:
        patience (int): The number of epochs to wait for improvement before
            stopping.
        min_pct_improvement (float): The minimum percent improvement in
            accuracy to continue training.
        counter (int): The number of epochs since the last improvement in
            accuracy.
        best_acc (float): The best validation accuracy observed so far.
        stop (bool): A flag indicating whether training should be stopped.
    """

    def __init__(self, patience=3, min_pct_improvement=3, verbose=False):
        """Initializes the EarlyStopping instance.

        Args:
            patience (int): The number of epochs to wait for improvement before
                stopping.
            min_pct_improvement (float): The minimum percent improvement in
                accuracy to continue training.
        """
        self.patience = patience
        self.min_pct_improvement = min_pct_improvement
        self.counter = 0
        self.best_acc = None
        self.stop = False
        self.verbose = verbose

    def __call__(self, curr_val_acc):
        """Updates the state of the EarlyStopping instance for a new epoch.

        Args:
            curr_val_acc (float): Validation accuracy for the current epoch.
        """
        if self.best_acc is None:
            self.best_acc = curr_val_acc
        elif curr_val_acc > self.best_acc:
            self.best_acc = curr_val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"\nEarlyStopping counter: {self.counter} out of "
                    f"{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

    def reset(self):
        """Resets the state of the EarlyStopping instance."""
        self.counter = 0
        self.best_acc = 0
        self.stop = False

def get_all_kmers(k):
    """
    generates all possible combinations of the letters a, g, t, and c of length k
    Examples:
        input: get_all_kmers(1)
        output: ['a', 'g', 't', 'c'] 
        input: get_all_kmers(2)
        output: ['aa', 'ag', 'at', 'ac', 'ga', 'gg', ... ]
    """
    all_kmers=[]
    for i in product('agtc', repeat = k):
        all_kmers.append(''.join(i))
    return all_kmers

def list_kmers(seq, k):
    """
    generates all kmers for the sequence (cap at 200 length sequence)
    Example
        input: list_kmers('agtcagtc', 2)
        output: ['ag', 'gt', 'tc', 'ca', 'ag', 'gt', 'tc']
    """
    kmers = []
    # Calculate how many kmers of length k there are
    if len(seq)<200:
        num_kmers=len(seq)-k+1
    else:
        #only for the first 200 letters of the sequence
        num_kmers = len(seq[:200]) - k + 1 
    # Loop over the kmer start positions
    for i in range(num_kmers):
        # Slice the string to get the kmer
        kmer = seq[i:i+k]
        kmers.append(kmer)
    return kmers

def create_feature_table(sequences,k):
    """
    creates a 2D array (or feature table) where each row corresponds to a sequence
    from the input list sequences, and each column corresponds to a possible k-mer
    of length k. The value at a specific row and column is the count of the
    corresponding k-mer in the corresponding sequence.
    """
    feature_table=[]
    for seq in sequences:
        cv = CountVectorizer(vocabulary=(get_all_kmers(k))) # lists counts of all words
        # representing sequences as sentences with words that are kmers i.e all kmers separated by space
        features=np.asarray(cv.fit_transform([(" ".join(list_kmers(seq,k)))]).toarray())  
        features=features.flatten().tolist()
        feature_table.append(features)
    return np.asarray(feature_table)