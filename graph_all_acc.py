# Graphs the validation accuracy over epochs

import matplotlib.pyplot as plt
import re
import numpy as np
import glob
import os

# Define a function to extract the number from the file name
def extract_number(file_path):
    print(file_path)
    # pattern = f'slum_outputs\\out.{job_ID}_([0-9]+).log'
    pattern = f'slurm_outputs\\\\out\\.{job_ID}_([0-9]+)\\.log'
    match = re.search(pattern, file_path)
    if match:
        print(f"Match! {int(match.group(1))}")
        return int(match.group(1))
    else:
        return 0

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
# with open('out.1855239.log', 'r') as file: # to find ratio of clst to sep
# with open('out.1855241.log', 'r') as file: # to find l1
# with open('out.1855240.log', 'r') as file: # to find ptype length
# with open('out.1857478.log', 'r') as file: # to find joint lrs after 2 pushes
# with open('out.1875496.log', 'r') as file: # to look back (do not save) 1842207? 
with open('out.1877399.log', 'r') as file: # to look back (do not save) 1842207? 
    data = file.read()

avg_last_25_epochs = {}

# job_ID = 1876793
# data = ""
# file_paths = sorted(glob.glob(f'slurm_outputs/out.{str(job_ID)}_*.log'), key=extract_number)
# print(file_paths)
# for file_path in file_paths:
#     with open(file_path, 'r') as file:
#         data += file.read()

def moving_average(a, n=15):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Split the data into different combinations
combinations = data.split('Attempting combination')
# print(len(combinations))
# print(combinations[34])
# Create the figure
plt.figure(figsize=(10, 6))

fancy = True

if not fancy:
    # NON FANCY  (for anaylsis)

    # new files start at combination 1. these loops should start at 1, they 
    # correspond directly with the combination number

    for i in range(1, len(combinations)):
    # for i in range(7, len(combinations)):
    # for i in range(1, 35, 7):
    # for i in range(1, 8):
    # for i in [4]:
        print(i)
        # print(combinations[i])
        # Extract the validation accuracies
        accs = re.findall('.*Val acc before epoch \d+: (\d+\.\d+)|.*Val acc at iteration \d+: (\d+\.\d+)', combinations[i])
        accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
        # Convert to floats
        accs = [float(acc) for acc in accs]
        # Apply moving average
        accs_smooth = moving_average(np.array(accs))
        if len(accs_smooth) >= 25:
            avg_last_25_epochs[i] = np.mean(accs_smooth[-25:])
        # Add to the graph
        plt.plot(accs_smooth, label=f'Combination {i}')

    # Add labels, title, and legend
    plt.title('Smoothed Accuracy for All Combinations')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.ylim(.925, 1)
    # plt.xlim(300, 500)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(avg_last_25_epochs.keys(), avg_last_25_epochs.values())
    plt.title('Average Accuracy for Last 25 Epochs for All Combinations')
    plt.xlabel('Combination')
    plt.ylabel('Average Validation Accuracy')
    plt.grid(True)
    plt.ylim(.925, 1)
    plt.show()

elif fancy:
    # FANCY VERSION (for use of creating graphs to put in the thesis)

    labels = [str(x) for x in range(1, 7)]
    # labels = [str(x) for x in range(1, 36, 2)]
    # labels = ['11', '17', '25', '29']
    # labels = [str(x) for x in range(11, 30, 2)]
    # labels = ['0', '5e-06', '1e-05', '5e-05', '0.0001', '0.0005']
    # labels = ['0.5', '0.1', '0.05', '0.01', '0.005', '0.001', '0.0005', '0.0001']
    # [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.95, 0.09, 0.85, 0.08, 0.75, 0.07, 0.65, 0.06, 0.55, 0.05]

    print(len(combinations))
    print(labels)
    print(len(labels))
    for i in range(1, len(combinations)):
    # for i in range(0,8):
    # for i in [1, 4, 8, 10]:
        print(f"i is {i}")
        # print(combinations[i])
        # Extract the validation accuracies
        accs = re.findall('.*Val acc before epoch \d+: (\d+\.\d+)|.*Val acc at iteration \d+: (\d+\.\d+)', combinations[i])
        accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
        # Convert to floats
        accs = [float(acc) for acc in accs]
        # Apply moving average
        accs_smooth = moving_average(np.array(accs))
        if len(accs_smooth) >= 25:
            avg_last_25_epochs[i] = np.mean(accs_smooth[-25:])
        # Add to the graph
        plt.plot(accs_smooth, label=labels[i-1])  # Use the corresponding label

    # Add labels, title, and legend
    plt.title('Comparing Prototype Lengths (Smoothed, n=20)')
    # plt.title('Comparing Prototype Lengths (Smoothed, n=20)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.legend()
    # Show the graph
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(avg_last_25_epochs.keys(), avg_last_25_epochs.values())
    plt.title('Accuracies When Starting Joint Training at Different Pushes')
    # plt.title('Accuracy for Different Prototype Lengths')
    plt.xlabel('Push Number')
    # plt.xlabel('Prototype Length')
    plt.ylabel('End Validation Accuracy')
    plt.grid(True)
    plt.ylim(.875, 1)
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