# Creates a bar graph of each combination vs. its value
# at some point during training (this can be set manually).

import matplotlib.pyplot as plt
import statistics
import re
import glob
import os

def average_chunks(input_list, chunk_size):
    # chunk size is the number of elements to average at a time
    result = [
        sum(input_list[i:i + chunk_size]) / len(input_list[i:i + chunk_size])
        for i in range(0, len(input_list), chunk_size)
    ]
    std_devs = [
        statistics.stdev(input_list[i:i + chunk_size])
        for i in range(0, len(input_list), chunk_size)
        if len(input_list[i:i + chunk_size]) > 1  # stdev requires at least two data points
    ]
    return result, std_devs


def extract_number(file_path):
    # Extract the number after the underscore and before .log using regex
    match = re.search(r'_(\d+)\.log$', file_path)
    return int(match.group(1)) if match else 0

# with open('out.1850026.log', 'r') as file:
# with open('out.1842252.log', 'r') as file: # clst vs sep
# with open('out.1855239.log', 'r') as file: # clst vs sep
#     data = file.read()

job_ID = 1878231 # latent weight, 3 iters: 1997591 #1877627 #1856524
data = ""
file_paths = glob.glob(os.path.join('slurm_outputs', f'out.{job_ID}_*.log'))
file_paths = sorted(file_paths, key=extract_number)
print(f"file_paths: {file_paths}")

for file_path in file_paths:
    print(f"file_path: {file_path}")
    with open(file_path, 'r') as file:
        data += file.read()

# Split the data into different combinations
combinations = data.split('Attempting combination')
print(f"len(combinations): {len(combinations)}")

# Initialize lists to store combination numbers and their corresponding accuracies
combination_numbers = []
accuracies = []
std_devs = []

# Iterate over the combinations
for i in range(1, len(combinations)):
# for i in range(1, 6):
    # Extract the validation accuracies
    # pattern = r'Val acc before epoch 70: (\d+\.\d+)'
    # pattern = r'\(Directly after push 2\) Val acc at iteration 0: (\d+\.\d+)'
    # pattern = r'\(Directly after push\) Val acc at iteration 0: (\d+\.\d+)'
    # pattern = f'Final validation accuracy before push: (\d+\.\d+)'
    # pattern = f'Final validation accuracy before push (\d): (\d+\.\d+)'
    # pattern = r'Test Accuracy: (\d*\.\d+)'
    pattern = r'Val acc at iteration 79: (\d*\.\d+)\nFinished search.'
    

    accs = re.findall(pattern, combinations[i])
    print(f"accs: {accs}")
    # Convert to floats
    accs = [float(acc) for acc in accs]
    # Add the combination number and its corresponding accuracy to the lists
    if accs:  # Check if accs is not empty
        combination_numbers.append(i)
        accuracies.append(accs[0])  # Use the first accuracy value for the bar graph

print(f"accuracies: {accuracies}")
avg_accs, std_devs = average_chunks(accuracies, 1)
print()
for acc in avg_accs:
    print(acc*100)
print()
for std in std_devs:
    print(std*100)
wait = input("pause")


# print(len(combination_numbers))
# print(len(accuracies))
# print(combination_numbers[0])
# print(accuracies[0])
# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(combination_numbers, avg_accs)
plt.grid(axis='y')

plt.ylim(min(avg_accs)-0.05, max(avg_accs)+0.05)

# Add labels and title
plt.title('Accuracy after Push 1')
# plt.title('Accuracy for All Combinations')
plt.xlabel('Combination Number')
plt.ylabel('Validation Accuracy')

plt.show()
