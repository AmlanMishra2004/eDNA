# (c) 2023 <name redacted for submission purposes>
# License: AGPLv3

"""Defines the dataset class for feeding augmented data to a network.

The three required functions for a PyTorch Dataset class are __init__() that
initializes any necessary variables, __getitem__() that is able to get a single
item from the dataset, and __len__() which gives the length of the dataset. In
__getitem__(), it augments the data before returning it. This is called online
augmentation, since it is performed when any data item is fetched, instead of
performing it all beforehand. This file was originally written by <name redacted for submission purposes>
(<name redacted for submission purposes>), but it has been heavily modified such that only the
skeleton of the original code remains. This version improves upon <name redacted for submission purposes>'s version
which assumed species to be the first column in the csv, sequence the second,
and that the species were grouped together. This version uses the column names
and doesn't assume species are grouped by rows.
"""

from collections import defaultdict

import torch
from torch.utils.data import Dataset

import utils

saved_sequences_prefix = './datasets/known_maine_species/'
labels_file = 'labels.npy'
labels_dict_file = 'labels_dict.npy'
sequence_file = 'sequences.npy'

class Sequence_Data(Dataset):
    """A class to hold and distribute sequence data for training or testing.

    This class is able to return a portion of self.sequences as a vector that
    is (batch_size x 4 x sequence_length). By specifying the parameters, an
    amount of noise can be added to the data before returning it.

    Attributes:
        X (pandas Series): The examples.
        y (pandas Series): The numerically-encoded labels for every sample.
        insertions: An array of length 2, inclusively describing the range of
            number of possible insertions in each sequence. For example, [0,2]
            would insert between 0 and 2 random insertions to each sequence.
        deletions: An array of length 2, inclusively describing the range of
            number of possible deletions in each sequence. For example, [1,1]
            would perform 1 random deletion for each sequence.
        mutation_rate: A decimal number that indicates the likelihood of a base
            being changed. This is applied to every bases in every sequence.
        encoding_mode: A string representing how to encode IUPAC ambiguity
            codes. 'probability' turns them into decimals. 'random' turns them
            into a random choice out of the possible base pairs. Other values 
            should not be used.
        seq_len: An integer representing the length of bases to which you want
            to truncate or pad all sequences.        
    """
    def __init__(self, X, y, insertions=[0,2], deletions=[0,2],
                 mutation_rate=0.05, encoding_mode='probability', seq_len=60):
        """Initializes the Sequence_Data instance."""

        if None in [insertions, deletions, mutation_rate,
                    encoding_mode, seq_len]:
            raise Exception("Please specify all arguments to create Dataset")
        
        # Copies X and y s.t. original dataframes/series are not modified.
        self.sequences = X.to_numpy()
        # self.labels = torch.tensor(y.values).long()
        self.labels = torch.tensor(y.values, dtype=torch.long)
        self.insertions = insertions
        self.deletions = deletions
        self.mutation_rate = mutation_rate
        self.encoding_mode = encoding_mode
        self.seq_len = seq_len

        # To verify some statistics about the dataset, uncomment below.

        # print(f"Shape of self.sequences: {self.sequences.shape}")
        # print(f"Shape of labels (should match len): {self.labels.shape}")
        # print("Created Dataset\n")

    def __getitem__(self,idx):
        """Returns a single pair of (example, label) given an index.

        First, adds random insertions for each sequence, then deletes random
        bases from each sequence. Then, mutates random bases in each sequence.
        Lastly, truncates or pads sequence to certain length, then returns the
        sequence turned into a vector using sequence_to_array(). Sequences are
        turned to arrays after augmentation because it is easier to augment
        strings rather than arrays.
        """

        seq = self.sequences[idx]
        label = self.labels[idx]

        # print(f"ORIGINAL SEQUENCE: \n{seq}")

        mutation_options = {'a':'cgt', 'c':'agt', 'g':'act', 't':'acg'}
        # Defaultdict returns 'acgt' if a key is not present in the dictionary.
        mutation_options = defaultdict(lambda: 'acgt', mutation_options)

        seq = utils.add_random_insertions(seq, self.insertions)
        seq = utils.apply_random_deletions(seq, self.deletions)
        seq = utils.apply_random_mutations(seq, self.mutation_rate,
                                           mutation_options)

        # print(f"MUTATED SEQUENCE: \n{seq}")

        # Pad or truncate to 60bp or 150bp. 'n' padding will be turned to
        # [0,0,0,0] in sequence_to_array().
        seq = seq.ljust(self.seq_len, 'n')[:self.seq_len]
        seq = utils.sequence_to_array(seq, self.encoding_mode)

        return torch.from_numpy(seq), label

    def __len__(self):
        """Returns the number of labels (presumably the number of examples)."""
        return self.labels.shape[0]
