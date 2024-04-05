# Graphs the validation accuracy over epochs

import matplotlib.pyplot as plt
import re
import numpy as np
import glob

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
# with open('out.1840698.log', 'r') as file: # to find last layer lr
# with open('out.1841514.log', 'r') as file: # to see if accuracy after push goes to 0 regardless of last layer lr
# with open('out.1842207.log', 'r') as file: # to find an even better (second round) warm lr
# with open('out.1842252.log', 'r') as file: # to view last layer lr results
# with open('out.1842260.log', 'r') as file: # to view last layer lr after second push
# with open('out.1843172.log', 'r') as file: # to see that learning rates need to change after each push
# with open('out.1845753.log', 'r') as file: # to find best last layer lr after 2 pushes
# KINDA RESET
# with open('out.1850021.log', 'r') as file: # to find latent weight
# with open('out.1850026.log', 'r') as file: # to find first layer lr and scheduler
# with open('out.1851380.log', 'r') as file: # to find first layer lr
# with open('out.1852559.log', 'r') as file: # to find ratio of clst to sep
with open('out.1845752.log', 'r') as file: # to look back (do not save) 1842207? 
    data = file.read()

job_ID = 1845752
data = ""
for filename in glob.glob(f'slurm_outputs/out.{job_ID}_*.log'):
    with open(filename, 'r') as file:
        data += file.read()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Split the data into different combinations
combinations = data.split('Attempting combination')
del combinations[0]
# Create the figure
plt.figure(figsize=(10, 6))

fancy = False

if not fancy:
    # NON FANCY  (for anaylsis)

    # new files start at combination 1. these loops should start at 1, they 
    # correspond directly with the combination number

    for i in range(1, len(combinations)+1):
    # for i in range(1, 10, 1):
    # for i in [4, 13, 22]:
        print(i)
        # Extract the validation accuracies
        accs = re.findall('.*Val acc before epoch \d+: (\d+\.\d+)|.*Val acc at iteration \d+: (\d+\.\d+)', combinations[i-1])
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
    plt.ylim(.40, 1)
    plt.show()

elif fancy:
    # FANCY VERSION (for use of creating graphs to put in the thesis)

    labels = ['0.5', '0.1', '0.05', '0.01', '0.005', '0.001', '0.0005', '0.0001']
    # [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.95, 0.09, 0.85, 0.08, 0.75, 0.07, 0.65, 0.06, 0.55, 0.05]

    print(len(combinations))
    print(len(labels))
    for i in range(1, len(combinations)+1):
    # for i in range(0,8):
    # for i in [0]:
        print(f"i is {i}")
        # print(combinations[i])
        # Extract the validation accuracies
        accs = re.findall('.*Val acc before epoch \d+: (\d+\.\d+)|.*Val acc at iteration \d+: (\d+\.\d+)', combinations[i])
        accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
        # Convert to floats
        accs = [float(acc) for acc in accs]
        # Apply moving average
        accs_smooth = moving_average(np.array(accs))
        # Add to the graph
        plt.plot(accs_smooth, label=labels[i])  # Use the corresponding label

    # Add labels, title, and legend
    plt.title('Comparing Warm Prototype Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.legend()
    # Show the graph
    plt.show()







# NEW VERSION that inorporates trials
    # # for i in range(1, len(combinations)+1):
    # for i in range(1,6):
    # # for i in [1, 8]:
    #     # Extract the validation accuracies
    #     accs = re.findall('.*Val acc before epoch \d+: (\d+\.\d+)|.*Val acc at iteration \d+: (\d+\.\d+)', combinations[i-1])
    #     accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
    #     # Convert to floats
    #     accs = [float(acc) for acc in accs]
    #     # Apply moving average
    #     accs_smooth = moving_average(np.array(accs))
    #     # Add to the graph
    #     plt.plot(accs_smooth, label=f'Combination {i}')