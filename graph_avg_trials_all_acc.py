# Idea: graph the average over several trials of the same hyperparameters
# abandoned 3/30/24 since it seems more productive to use existing code than to debug this file

from collections import defaultdict
import matplotlib.pyplot as plt
import re
import numpy as np

# Create a dictionary to store the accuracies of each trial
trial_accs = defaultdict(list)

with open('out.1838072.log', 'r') as file:
    data = file.read()

# Split the data into different combinations
combinations = data.split('Attempting combination')

# Create the figure
plt.figure(figsize=(10, 6))

def moving_average(a, n=1) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# For each combination
for i in range(1, len(combinations)):
    # Extract the trial number
    trial_number = int(re.findall('Trial (\\d+)', combinations[i])[0])
    
    # Extract the validation accuracies
    accs = re.findall('.*Val acc before epoch \\d+: (\\d+\\.\\d+)|.*Val acc at iteration \\d+: (\\d+\\.\\d+)', combinations[i])
    accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
    
    # Convert to floats
    accs = [float(acc) for acc in accs]
    
    # Apply moving average
    accs_smooth = moving_average(np.array(accs))
    
    # Add to the dictionary
    trial_accs[trial_number].append(accs_smooth)

# Create a new dictionary to store the averaged accuracies
averaged_accs = {}

# For each trial
for trial, accs in trial_accs.items():
    # Calculate the average
    averaged_accs[trial] = sum(accs) / len(accs)

# Create the figure
plt.figure(figsize=(10, 6))

# For each trial
for trial, accs in averaged_accs.items():
    # Add to the graph
    plt.plot(accs, label=f'Trial {trial}')

# Add labels, title, and legend
plt.title('Smoothed Accuracy for All Trials')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.legend()
plt.show()
