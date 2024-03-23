# Graphs the last layer accuracy over epochs

import matplotlib.pyplot as plt
import re

# Open the file
# with open('out.1831432.log', 'r') as file: # 9
# with open('out.1832475.log', 'r') as file: # the single best of the 9
# with open('out.1832553.log ', 'r') as file: # the single best, looking at push_gap
# with open('out.1832676.log', 'r') as file: # 10 pushes

# with open('out.1838072.log', 'r') as file: # fixed initialization, to find warm lr
# with open('out.1838398.log', 'r') as file: # to find last layer epochs and lr
# with open('out.1838491.log', 'r') as file: # to find last layer epochs and lr
    data = file.read()

# Split the data into different combinations
combinations = data.split('Attempting combination')

# For each combination
for i in range(1, len(combinations)):
    # Extract the validation accuracies
    train_accs = re.findall('Train acc at iteration \d+: (\d+\.\d+)', combinations[i])

    # Convert to floats
    train_accs = [float(acc) for acc in train_accs]

    # Generate the graph
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs)
    plt.title(f'Last Layer Training Accuracy for Combination {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.grid(True)
    plt.show()

