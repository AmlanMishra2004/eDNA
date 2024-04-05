# Creates a bar graph of each combination vs. its value
# at some point during training (this can be set manually).

import matplotlib.pyplot as plt
import re

# with open('out.1850026.log', 'r') as file:
with open('out.1842252.log', 'r') as file: # clst vs sep
    data = file.read()

# Split the data into different combinations
combinations = data.split('Attempting combination')
del combinations[0]

# Initialize lists to store combination numbers and their corresponding accuracies
combination_numbers = []
accuracies = []

# Iterate over the combinations
for i in range(1, len(combinations)+1):
# for i in range(1, 6):
    # Extract the validation accuracies
    # pattern = r'Val acc before epoch 70: (\d+\.\d+)'
    # pattern = r'\(Directly after push 2\) Val acc at iteration 0: (\d+\.\d+)'
    pattern = r'\(Directly after push\) Val acc at iteration 0: (\d+\.\d+)'
    accs = re.findall(pattern, combinations[i-1])
    # Convert to floats
    accs = [float(acc) for acc in accs]
    # Add the combination number and its corresponding accuracy to the lists
    if accs:  # Check if accs is not empty
        combination_numbers.append(i)
        accuracies.append(accs[0])  # Use the first accuracy value for the bar graph
print(len(combination_numbers))
print(len(accuracies))
print(combination_numbers[0])
print(accuracies[0])
# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(combination_numbers, accuracies)
plt.grid(axis='y')

plt.ylim(min(accuracies)-0.05, max(accuracies)+0.05)

# Add labels and title
plt.title('Accuracy for All Combinations')
plt.xlabel('Combination Number')
plt.ylabel('Validation Accuracy')

plt.show()
