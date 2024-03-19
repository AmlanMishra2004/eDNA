import matplotlib.pyplot as plt
import re

# Open the file
# with open('out.1831432.log', 'r') as file: # 9
with open('out.1832475.log', 'r') as file: # just one
    data = file.read()

# Split the data into different combinations
combinations = data.split('Attempting combination')

# For each combination
for i in range(1, len(combinations)):
    # Extract the validation accuracies
    val_accs = re.findall('Val acc before epoch \d+: (\d+\.\d+)', combinations[i])

    # Convert to floats
    val_accs = [float(acc) for acc in val_accs]

    # Generate the graph
    plt.figure(figsize=(10, 6))
    plt.plot(val_accs)
    plt.title(f'Validation Accuracy for Combination {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.show()

