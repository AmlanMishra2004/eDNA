# Graphs the validation accuracy over epochs

import matplotlib.pyplot as plt
import re

# Open the file
# with open('out.1831432.log', 'r') as file: # 9
# with open('out.1832475.log', 'r') as file: # the single best of the 9
# with open('out.1832553.log ', 'r') as file: # the single best, looking at push_gap
# with open('out.1832676.log', 'r') as file: # 10 pushes

# with open('out.1838072.log', 'r') as file: # fixed initialization, to find warm lr
# with open('out.1838398.log', 'r') as file: # to find last layer epochs and lr
# with open('out.1838484.log', 'r') as file: # to find warm lr scheduler
# with open('out.1838624.log', 'r') as file: # to find push gap
with open('out.1838708.log', 'r') as file: # to find push gap
    data = file.read()

# Split the data into different combinations
combinations = data.split('Attempting combination')

# For each combination
for i in range(1, len(combinations)):
    # Extract the validation accuracies
    accs = re.findall('Val acc before epoch \d+: (\d+\.\d+)|Train acc at iteration \d+: (\d+\.\d+)', combinations[i])
    accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
    # Convert to floats
    accs = [float(acc) for acc in accs]

    # Generate the graph
    plt.figure(figsize=(10, 6))
    plt.plot(accs)
    plt.title(f'Accuracy for Combination {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation OR Train Accuracy')
    plt.grid(True)
    plt.show()

