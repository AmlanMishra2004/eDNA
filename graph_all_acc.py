# Graphs the validation accuracy over epochs

import matplotlib.pyplot as plt
import re
import numpy as np

# Open the file
# with open('out.1831432.log', 'r') as file: # 9
# with open('out.1832475.log', 'r') as file: # the single best of the 9
# with open('out.1832553.log ', 'r') as file: # the single best, looking at push_gap
# with open('out.1832676.log', 'r') as file: # 10 pushes

# with open('out.1838072.log', 'r') as file: # fixed initialization, to find warm lr
# with open('out.1838398.log', 'r') as file: # to find last layer epochs and lr FROM SCRATCH, without warm training first
# with open('out.1838484.log', 'r') as file: # to find warm lr scheduler
# with open('out.1838624.log', 'r') as file: # to find push gap
# with open('out.1838708.log', 'r') as file: # to find push gap -> 35 
# with open('out.1838794.log', 'r') as file: # to see if push gap should change -> too many epochs, can't see
# FIXED WARM LR AFTER PUSH
# with open('out.1840139.log', 'r') as file: # to find push_gap

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# with open('out.1840698.log', 'r') as file: # to find last layer lr
# with open('out.1841514.log', 'r') as file: # to see if accuracy after push goes to 0 regardless of last layer lr
# with open('out.1842207.log', 'r') as file: # to find an even better (second round) warm lr
# with open('out.1842252.log', 'r') as file: # to view last layer lr results
with open('out.1842260.log', 'r') as file: # to view last layer lr after second push
# with open('out.1840698.log', 'r') as file: # to look back (do not save)
    data = file.read()

# Split the data into different combinations
combinations = data.split('Attempting combination')

# Create the figure
plt.figure(figsize=(10, 6))

# For each combination
# for i in range(1, len(combinations)):
# for i in range(4, 9):
for i in [5,6]:
    # Extract the validation accuracies
    accs = re.findall('.*Val acc before epoch \d+: (\d+\.\d+)|.*Val acc at iteration \d+: (\d+\.\d+)', combinations[i])
    accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
    # Convert to floats
    accs = [float(acc) for acc in accs]
    # Apply moving average
    accs_smooth = moving_average(np.array(accs))

    # Add to the graph
    plt.plot(accs_smooth, label=f'Combination {i}')

# Add labels, title, and legend
plt.title('Smoothed Accuracy for All Combinations')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.legend()

# Show the graph
plt.show()


# with open('out.1840698.log', 'r') as file: # to find last layer lr
#     data = file.read()

# # Split the data into different combinations
# combinations = data.split('Attempting combination')

# # For each combination
# for i in range(1, len(combinations)):
#     # Extract the validation accuracies
#     accs = re.findall('.*Val acc before epoch \d+: (\d+\.\d+)|.*Val acc at iteration \d+: (\d+\.\d+)', combinations[i])
#     accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
#     # Convert to floats
#     accs = [float(acc) for acc in accs]

#     # Generate the graph
#     plt.figure(figsize=(10, 6))
#     plt.plot(accs)
#     plt.title(f'Accuracy for Combination {i}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Validation Accuracy')
#     plt.grid(True)
#     plt.show()

