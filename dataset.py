import csv
import torch
import os
import numpy as np
from torch.utils.data import Dataset

saved_sequences_prefix = './datasets/known_maine_species/'
labels_file = 'labels.npy'
labels_dict_file = 'labels_dict.npy'
sequence_file = 'sequences.npy'

class Sequence_Data(Dataset):
    def __init__(self,
                data_path='./datasets/Data Prep_ novel species - ood dataset of maine genus.csv',
                target_sequence_length=250,
                sequence_count_threshold=2,
                transform=None):
        # It's easier to work with the tsv version, since there are some commas
        # used in it that mess everything up for the csv conversion
        print("Using dataset: {}".format(data_path))
        self.data_path = data_path
        # The length to restrict sequeneces to
        self.target_sequence_length = target_sequence_length

        self.label_dict = {}
        self.current_max_label = 0
        self.len = 0
        self.sequence_count_threshold = sequence_count_threshold
        self.transform = transform

        print("Initializing data...")
        with open(data_path) as csv_data:
            csv_file = list(csv.reader(csv_data, delimiter=','))

            self.len = len(csv_file) - 1

            # self.sequences is a tensor of shape num_sequences x 4 x sequence_length
            self.sequences = torch.zeros(self.len, 4, self.target_sequence_length)
            self.labels = torch.zeros(self.len)
            species1=''
            species_counter=0
            drop_species_offset=0
            for index, row in enumerate(csv_file):
                if index == 0:
                    header_row = row
                else:
                    if species1==row[0]:
                        species_counter+=1
                    else:
                        if species_counter < self.sequence_count_threshold:
                            for subtractor in range(species_counter):
                                self.sequences[index-subtractor-1-drop_species_offset]=0
                            drop_species_offset+=species_counter
                        species_counter=0
                    # Update the label dictionary to include this class if it doesn't
                    if row[0] not in self.label_dict.keys():
                        self.label_dict[row[0]] = self.current_max_label
                        self.current_max_label += 1

                    self.labels[index-1-drop_species_offset] = int(self.label_dict[row[0]])
                    if len(row[1]) < self.target_sequence_length:
                        self.sequences[index-1-drop_species_offset, :, :len(row[1])] = torch.Tensor(self.sequence_to_array(row[1])[:, :])
                    else:
                        self.sequences[index-1-drop_species_offset] = torch.Tensor(self.sequence_to_array(row[1])[:, :target_sequence_length])
                    species1=row[0]


        print("Total dataset size is %d" % (self.len))

    def __getitem__(self,idx):
        item = self.sequences[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            item = self.transform(item)
        return item, label
  
    def __len__(self):
        return self.len
        
    '''
    Converts an array of DNA bases to a 4 channel numpy array, 
    with A -> channel 0, T -> channel 1, C -> channel 2, and G -> channel 3,
    assigning fractional values for ambiguity codes. A full list of these 
    codes can be found at https://droog.gs.washington.edu/mdecode/images/iupac.html 
    Input: String of bases
    Output: 4 * str_len array of integers
    '''
    def sequence_to_array(self, bp_sequence):
        output_array = torch.zeros(4, len(bp_sequence))

        for i, letter in enumerate(bp_sequence):
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
                print("ERROR: Unknown base '{}' encountered at index {} in {}".format(letter, i, bp_sequence))

        return output_array

if __name__ == '__main__':
    dataset = Sequence_Data() #dataloader
    from torch.utils.data import DataLoader 
    train_loader = DataLoader(dataset,shuffle=True,batch_size=1)
    for item, label in train_loader:
        print("label: ", label)
        print("For sequence: ", item)