import csv
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import math

class Sequence_Data_Alt(Dataset):
    def __init__(self, data_path="test.csv", 
                        target_sequence_length=150,
                        predefined_label_dict={},
                        augment=None):
        self.target_sequence_length = target_sequence_length
        self.augment = augment

        with open(data_path) as csv_data:
            csv_rows = csv.reader(csv_data, delimiter=',')
            # Subtract 1 to deal with the header row
            self.num_samples = len(list(csv_rows))
            csv_data.seek(0)

            # Allocating with 0 as the default since we want padding
            self.X = torch.zeros(self.num_samples, 4, self.target_sequence_length)
            self.y = torch.zeros(self.num_samples)

            # Should look like {'A': 0, 'B': 1, 'C': 2}
            self.label_dict = predefined_label_dict
            if len(self.label_dict.keys()) > 0:
                self.next_class_label = max(self.label_dict.values()) + 1
            else:
                self.next_class_label = 0

            i = 0
            for row in csv_rows:
                # Skip the header row
                if i == 0:
                    i += 1
                    continue

                if len(row[1]) < self.target_sequence_length:
                    self.X[i, :, :len(row[1])] = self.sequence_to_array(row[1])
                else:
                    self.X[i] = self.sequence_to_array(row[1][:self.target_sequence_length])

                if row[0] not in self.label_dict.keys():
                    self.label_dict[row[0]] = self.next_class_label
                    self.next_class_label += 1

                self.y[i] = self.label_dict[row[0]]
                i += 1

        print("self.num_samples")
        print(self.num_samples)
        print("self.next_class_label")
        print(self.next_class_label)
        print("self.label_dict")
        print(self.label_dict)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.X[idx]
        if not (self.augment is None):
            item = self.augment(item)
        label = int(self.y[idx])
        return item, label

    def get_count_by_class(self):
        count_vector = torch.zeros(self.next_class_label)
        for i in range(self.num_samples):
            _, label = self.__getitem__(i)
            count_vector[label] += 1
        print(count_vector)
        return count_vector

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