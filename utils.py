'''
This file contains functions that are used by evaluate_model.py.
'''

import pandas as pd
import random
from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



# PREPROCESSING ---------------------------------------------------------------
    
'''
From Zurich
Returns a dataframe that contains the reverse complement of every sequence,
in addition to all of the original sequences.
'''
def add_reverse_complements(data, seq_col, comp):
    data_rev = data.copy()
    data_rev[seq_col] = data_rev[seq_col].apply(lambda x: ''.join(map(lambda y: comp[y], x[::-1]))) 
    new_data = pd.concat([data, data_rev])
    return new_data

'''
Remove any sequences that belong to species who have fewer than
seq_count_thresh sequences.
'''
def remove_species_with_too_few_sequences(df, species_col, seq_count_thresh):
    orig_num_rows = df.shape[0]
    species_counts = df[species_col].value_counts()
    valid_species = species_counts[species_counts >= seq_count_thresh].index
    df = df[df[species_col].isin(valid_species)]
    print(f"Removed {orig_num_rows - df.shape[0]} rows while enforcing "
            f"the {seq_count_thresh} sequence threshold per species.")
    return df


'''
Unused
Cap sequence length at some number, filling in shorter sequences with 'z'
These z characters will be turned into [0,0,0,0] later by sequence_to_array()
as part of Dataset.__getitem__().
'''
def make_sequences_uniform_length(df, seq_col, length):
    df[seq_col] = df[seq_col].map(lambda x: x[:length] if len(x) > length else x.ljust(length, 'z'))
    return df

'''
From Zurich
Returns a dataframe in which there is an equal number of sequences from every
species. Each species has the same number of sequences as the maximum number of
sequences for a single species. Species are upsampled by randomly duplicating
a sequence.
'''
def oversample_underrepresented_species(df, species_col):
    value_counts = df[species_col].value_counts()
    seq_counts = value_counts.to_dict()
    max_count = max(value_counts)
    new_ind = []

    for val in seq_counts.keys():
        ind = [i for i, j in enumerate(df[species_col]) if j == val]
        new_ind.extend(ind)
        k = round(max_count - seq_counts[val])
        new_ind.extend(random.choices(ind, k=k))

    oversample = df.iloc[new_ind]
    return oversample

'''
Splits data into train and test, stratified on a given column.
Will keep at least one of each species in train and test. This may result in
a different ratio of train/test than the supplied ratio.
'''
def stratified_split(data, column, ratio):
    to_keep = []
    to_hold = []
    # get each class and its frequency
    counts = data[column].value_counts()
    for t in counts.items():
        taxa_list = data[data[column]==t[0]][column].index.to_list()
        random.shuffle(taxa_list)
        split = max(1, round(t[1]*ratio))
        to_hold.extend(taxa_list[:split])
        to_keep.extend(taxa_list[split:])
    to_keep = data.loc[to_keep]
    to_hold = data.loc[to_hold]
    return to_keep, to_hold

def print_descriptive_stats(data, cols):
    print('-' * 60)
    print(f"{'Number of sequences:':<50} {len(data):>10}")
    for col in cols:
        num_distinct = data[col].nunique()
        print(f"{'Number of distinct ' + col:<50} {num_distinct:>10}")

        avg_sequences = round(len(data) / num_distinct, 2)
        print(f"{'Average number of sequences per ' + col:<50} {avg_sequences:>10.2f}")

        value_counts = data[col].value_counts()
        range_sequences = value_counts.max() - value_counts.min()
        print(f"{'Range in the number of sequences per ' + col:<50} {range_sequences:>10}")
    print('-' * 60)


'''
Converts an array of DNA bases to a 4 channel numpy array, 
with A -> channel 0, T -> channel 1, C -> channel 2, and G -> channel 3,
assigning fractional values for ambiguity codes. A full list of these 
codes can be found at https://droog.gs.washington.edu/mdecode/images/iupac.html 
or https://www.dnabaser.com/articles/IUPAC%20ambiguity%20codes.html.

Input:
sequence: string of bases, lowercase or uppercase
mode: 'probability' or 'random', how it deals with IUPAC ambiguity codes
    - probability turns them into decimals
    - random turns it into a random choice out of the possible base pairs

Output:
4 x str_len array
    Example: 
    dataset.sequence_to_array('atcg', 'probability') would return
    [[1 0 0 0]
    [0 1 0 0]
    [0 0 0 1]
    [0 0 1 0]]
    ...where the associated base pairs are the nth entries of every vector.
'''
def sequence_to_array(sequence, mode):
    if mode == 'probability':
        mapping = {
            'a':[1, 0, 0, 0],
            't':[0, 1, 0, 0],
            'u':[0, 1, 0, 0], # u = t
            'c':[0, 0, 1, 0],
            'g':[0, 0, 0, 1],
            # IUPAC ambiguity codes: https://www.dnabaser.com/articles/IUPAC%20ambiguity%20codes.html
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
        mapping = defaultdict(lambda: [0,0,0,0], mapping)

        vector = []
        for base in sequence.lower():
            vector.append(mapping[base])
            # try:
            #     vector.append(mapping[base])
            # except KeyError:
            #     if base == ' ':
            #         print(f"Warning: Unknown base '{base}' not found in mapping for sequence {sequence}. Using [0, 0, 0, 0].")
            #     vector.append([0, 0, 0, 0])
        return np.transpose(np.array(vector)).astype(np.float32) # float64 -> float32 for gpu speed
    
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
            elif letter == 'N':
                # N indicates complete uncertainty
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
                print("ERROR: Unknown base '{}' encountered at index {} in {}".format(letter, i, sequence))

        print("SHAPE\n\n\n")
        print(output_array.shape)
        return output_array




# END PREPROCESSING -----------------------------------------------------------

# Takes a single sequence as a string and adds random insertions
# ex. if insertions = [0,2], inserts between 0 and 2 random base pairs
def add_random_insertions(seq, insertions):
    insertions = random.randint(insertions[0],insertions[1])
    for i in range(insertions):
        pos = random.sample(range(len(seq) + 1), 1)[0]
        seq = seq[0:pos] + random.sample(['acgt'], 1)[0] + seq[pos:]
    return seq

# Takes a single sequence as a string and deletes random base pairs
# ex. if deletions = [0,2], deletes between 0 and 2 random base pairs
def apply_random_deletions(seq, deletions):
    deletions = random.randint(deletions[0], deletions[1])
    for i in range(deletions):
        pos = random.sample(range(len(seq) + 1), 1)[0]
        seq = seq[0:pos] + seq[pos+1:]
    return seq

# Takes a single sequence as a string and adds random mutations
# Every base pair in the string has a mutation_rate chance (as a decimal) of 
#   switching to some other base pair, as defined in mutation_options.
def apply_random_mutations(seq, mutation_rate, mutation_options):
    # generate a list of random nums [0-1], the same length of the sequence
    mutations = np.random.random(len(seq))
    for idx,char in enumerate(seq):
        if mutations[idx] < mutation_rate:
            seq = seq[0:idx] + random.sample(mutation_options[char], 1)[0] + seq[idx+1:]
    return seq

'''
Turns a given dataframe into torch or tf tensors

Returns (sequences, labels)
For pytorch, sequences is an array of shape (num_sequences, channels=4, length=60)
For tensorflow, sequences is an array of shape (num_sequences, height=1, width=60, channels=4)

df contains about 11,000 rows. Each row contains an array that is of shape (4, 60)
'''
def encode_all_data(df, seq_len, seq_col, species_col, encoding_mode,
                    include_height, format, mutate, insertions, deletions,
                    mutation_rate, iupac):
    if mutate:
        mutation_options = {'a':'cgt', 'c':'agt', 'g':'act', 't':'acg'}
        # defaultdict returns 'acgt' if a key is not present in the dictionary
        mutation_options = defaultdict(lambda: 'acgt', mutation_options)

        df[seq_col] = df[seq_col].map(lambda x: add_random_insertions(x, insertions))
        df[seq_col] = df[seq_col].map(lambda x: apply_random_deletions(x, deletions))
        df[seq_col] = df[seq_col].map(lambda x: apply_random_mutations(x, mutation_rate, mutation_options))

    # Pad or truncate to 60bp. 'z' padding will be turned to [0,0,0,0] below
    df[seq_col] = df[seq_col].apply(lambda seq: seq.ljust(60, 'z')[:60])

    # Turn every base pair character into a vector
    df[seq_col] = df[seq_col].map(lambda x: sequence_to_array(x, encoding_mode))

    sequences = np.array(df[seq_col].tolist())

    # pd.set_option('display.max_colwidth', None)
    # print(f"FIRST ELE in DF: {(df[seq_col].head(1))}")
    # print(sequences.shape)
    # print(f"SEQUENCES[0]=\n\n{sequences[0]}\n\n")

    '''
    if format == "torch":
        sequences = torch.from_numpy(sequences) # turn to torch tensor
        labels = torch.tensor(df[species_col].values) # float32 by default
        return sequences, labels
    
    elif format == "tf":
        # Convert to (11269, 4, 60) to (11269, 1, 60, 4)
        new_sequences = np.zeros((sequences.shape[0], 1, 60, 4)).astype(np.float32)
        for i in range(sequences.shape[0]):
            new_sequences[i] = sequences[i].T
        new_sequences = tf.convert_to_tensor(new_sequences)
        labels = tf.convert_to_tensor(df[species_col].values)
        # print(f"NEW SEQUENCES[0]=\n\n{new_sequences[0]}\n\n")
        # print(f"NEW SEQUENCES[0] shape=\n{new_sequences[0].shape}\n\n")
        # print(f"NEW SEQUENCES[-1]=\n\n{new_sequences[-1]}\n\n")
        # print(f"NEW SEQUENCES[-1] shape=\n{new_sequences[-1].shape}\n\n")
        # print(new_sequences.shape)
        # print(labels.shape)
        # pause = input("FINAL PAUSE")
        return new_sequences, labels
    elif format == "np":
        # Convert to (11269, 4, 60) to (11269, 1, 60, 4)
        new_sequences = np.zeros((sequences.shape[0], 1, 60, 4)).astype(np.float32)
        for i in range(sequences.shape[0]):
            new_sequences[i] = sequences[i].T
        labels = df[species_col].to_numpy()
        return new_sequences, labels
    else:
        raise ValueError("Framework name is incorrect. Use 'torch' or 'tf'.")
    '''
    if format == "df":
        return df

    if include_height:
        if format == "torch":
            raise ValueError("Include height with torch is not configured")
        elif format == "tf":
            # Convert to (11269, 4, 60) to (11269, 1, 60, 4)
            new_sequences = np.zeros((sequences.shape[0], 1, 60, 4)).astype(np.float32)
            for i in range(sequences.shape[0]):
                new_sequences[i] = sequences[i].T
            new_sequences = tf.convert_to_tensor(new_sequences)
            labels = tf.convert_to_tensor(df[species_col].values)
            # print(f"NEW SEQUENCES[0]=\n\n{new_sequences[0]}\n\n")
            # print(f"NEW SEQUENCES[0] shape=\n{new_sequences[0].shape}\n\n")
            # print(f"NEW SEQUENCES[-1]=\n\n{new_sequences[-1]}\n\n")
            # print(f"NEW SEQUENCES[-1] shape=\n{new_sequences[-1].shape}\n\n")
            # print(new_sequences.shape)
            # print(labels.shape)
            # pause = input("FINAL PAUSE")
            return new_sequences, labels
        elif format == "np":
            # Convert to (11269, 4, 60) to (11269, 1, 60, 4)
            new_sequences = np.zeros((sequences.shape[0], 1, 60, 4)).astype(np.float32)
            for i in range(sequences.shape[0]):
                new_sequences[i] = sequences[i].T
            labels = df[species_col].to_numpy()
            return new_sequences, labels
        else:
            raise ValueError("Framework name is incorrect. Use 'torch' or 'tf' or 'np'.")
    elif not include_height:
        if format == "torch":
            sequences = torch.from_numpy(sequences) # turn to torch tensor
            labels = torch.tensor(df[species_col].values) # float32 by default
            return sequences, labels
        elif format == "tf":
            # Convert to (11269, 4, 60) to (11269, 60, 4)
            new_sequences = np.zeros((sequences.shape[0], 60, 4)).astype(np.float32)
            for i in range(sequences.shape[0]):
                new_sequences[i] = sequences[i].T
            new_sequences = tf.convert_to_tensor(new_sequences)
            labels = tf.convert_to_tensor(df[species_col].values)
            return new_sequences, labels
        elif format == "np":
            # Convert to (11269, 4, 60) to (11269, 60, 4)
            new_sequences = np.zeros((sequences.shape[0], 60, 4)).astype(np.float32)
            for i in range(sequences.shape[0]):
                new_sequences[i] = sequences[i].T
            labels = df[species_col].to_numpy()
            return new_sequences, labels
        else:
            raise ValueError("Framework name is incorrect. Use 'torch' or 'tf' or 'np'.")


# Unfinished
def graph_train_vs_test_acc(train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
