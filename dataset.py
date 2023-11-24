import torch
import numpy as np
import pandas as pd
import random
import utils
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 


saved_sequences_prefix = './datasets/known_maine_species/'
labels_file = 'labels.npy'
labels_dict_file = 'labels_dict.npy'
sequence_file = 'sequences.npy'

"""
Input
data: (dataframe) that includes all columns
seq_col: the (string) name of the column containing the sequences
species_col: the (string) name of the column containing the species labels
insertions: an (array) of length 2, inclusively describing the range of number
    of insertions in each sequence.
    ex. [0,2] would insert between 0 and 2 random insertions to each sequence
deletions: same as insertions, except describes the possible range of number of
    base pairs that will be deleted for each sequence.
mutation_rate: the decimal number that indicates the likelihood of a base pair
    being flipped. This is applied to all base pairs in all sequences.
    ex. 0.05 would mean there is a 5% chance A is replaced with T.
encoding_mode: (string) how to deal with IUPAC ambiguity codes
    - 'probability' turns them into decimals
    - 'random' turns it into a random choice out of the possible base pairs
iupac: a (dict) containing every base pair and its opposite, ex. a:t, s:w
seq_len: the (int) length to which you want to cap/pad all sequences

Output
- able to return a portion of self.sequences as a vector that is
    (num_sequences x 4 x sequence_length)

Improvements:
- Jon's version assumed species to be the first column in the csv, sequence the second
- Jon's version assumed that the species were grouped together
"""
class Sequence_Data(Dataset):
    def __init__(self, data, seq_col, species_col, insertions, deletions,
                 mutation_rate, encoding_mode, iupac, seq_len):
        
        # Ensure all agruments are supplied
        if None in [seq_col, species_col, insertions, deletions,
                    mutation_rate, encoding_mode, iupac]:
            raise Exception("Please specify all arguments when creating Dataset")
        
        # 'data' is used to keep all of the columns, but it is not used later
        #   since only seq and species are used, and they are assigned to 
        #   different variables below.
        self.data = data
        # sequences are turned to nparrays *after* mutation, since they are
        #   manipulated as strings anyway when buffering to 60 or 150 chars
        self.sequences = data[seq_col].to_numpy()
        self.labels = torch.tensor(data[species_col].values).long()
        self.seq_col = seq_col
        self.species_col = species_col
        self.insertions = insertions
        self.deletions = deletions
        self.mutation_rate = mutation_rate
        self.encoding_mode = encoding_mode
        self.iupac = iupac
        self.seq_len = seq_len

        # print(f"Shape of self.sequences: {self.sequences.shape}")
        # print(f"Number of sequences: {len(self.sequences)}")
        # print(f"Length of labels (should match): {len(self.labels)}")
        # print("Created Dataset\n")

    '''
    Required function for Dataset that gets a single item given an index.
    - adds random insertions for each sequence
    - deletes random bp from each sequence
    - mutates random bp in each sequence
    - truncates or pads sequence to certain length (60bp)
    - returns the sequence turned into a vector using sequence_to_array()
    '''
    def __getitem__(self,idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        # print(f"ORIGINAL SEQUENCE: \n{seq}")

        mutation_options = {'a':'cgt', 'c':'agt', 'g':'act', 't':'acg'}
        # defaultdict returns 'acgt' if a key is not present in the dictionary
        mutation_options = defaultdict(lambda: 'acgt', mutation_options)

        seq = utils.add_random_insertions(seq, self.insertions)
        seq = utils.apply_random_deletions(seq, self.deletions)
        seq = utils.apply_random_mutations(seq, self.mutation_rate, mutation_options)

        # print(f"MUTATED SEQUENCE: \n{seq}")

        # Pad or truncate to 60bp or 150bp. 'z' padding will be turned to [0,0,0,0] below
        seq = seq.ljust(self.seq_len, 'z')[:self.seq_len]

        # Turn every base pair character into a vector
        seq = utils.sequence_to_array(seq, self.encoding_mode)

        return torch.from_numpy(seq), label

    '''
    Required function for Dataset that returns the number of examples in the set.
    '''
    def __len__(self):
        return self.labels.shape[0]
