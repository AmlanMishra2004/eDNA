import glob
import re
import matplotlib.pyplot as plt

def extract_run_number_from_file(file_path):
    # print(file_path)
    pattern = f'slurm_outputs\\\\out\\.{job_ID}_([0-9]+)\\.log'
    match = re.search(pattern, file_path)
    if match:
        return int(match.group(1))
    else:
        return 0

def extract_test_acc_from_file(file):
    content = file.read()
    matches = re.findall(r'Test Accuracy: (\d+\.\d+)', content)
    if matches:
        accuracies = [float(match) for match in matches]
        average_accuracy = sum(accuracies) / len(accuracies)
        return average_accuracy
    else:
        raise ValueError(f"Could not get test accuracy from file {file.name}")

################################
job_ID = 2041209 #1997515
x_label = "Prototype Length"
y_label = 'Test Accuracy'
################################

test_accs = []
file_paths = sorted(glob.glob(f'slurm_outputs/out.{str(job_ID)}_*.log'), key=extract_run_number_from_file)

for file_path in file_paths:
    print(f"Searching file path: {file_path}")
    with open(file_path, 'r') as file:
        test_acc = extract_test_acc_from_file(file)
        if test_acc is not None:
            test_accs.append(test_acc)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(range(len(test_accs)), test_accs, color='#1f77b4')

# Add labels and title
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title('Test Accuracies for Different Runs')
labels = [f'{i*2-1}' for i in range(1,len(test_accs)+1)]
labels = [f'{i}' for i in range(1,len(test_accs)+1)]
# labels = [15, 20, 25, 35, 40, 45, 50, 55, 60, 65, 75, 80, 85, 95, 100]
print(f"Labels: {labels}")
plt.xticks(range(len(test_accs)), labels)
plt.grid(True)
ax = plt.gca()  # Get the current Axes instance
ax.xaxis.grid(False)
plt.ylim(.9, 1)

# Show the plot
plt.show()





# labels = [str(x) for x in range(1, 36, 2)] # all prototype lengths
# # labels = [format(x*0.1, '.1f') for x in range(0, 11)]
# # labels = ['0', '0.2', '0.4', '0.6', '0.8', '1', '-1', '-1', '-1']
# # labels = [str(x) for x in range(1, 36, 2)]
# # labels = [str(x) for x in range(1, 22, 2)]
# # labels = ['11', '17', '25', '29']
# # labels = [str(x) for x in range(11, 30, 2)]
# # labels = ['0', '5e-06', '1e-05', '5e-05', '0.0001', '0.0005']
# # labels = ['0.5', '0.1', '0.05', '0.01', '0.005', '0.001', '0.0005', '0.0001']
# # [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.95, 0.09, 0.85, 0.08, 0.75, 0.07, 0.65, 0.06, 0.55, 0.05]

# print(f"Labels: {labels}")

# print(labels)
# print(len(labels))
# # for i in range(1, len(combinations)):
# # for i in [0, 2, 4, 6, 8, 10]:
# for i in [1, 3, 5, 7, 9, 11]:
# # for i in range(0,8):
# # for i in [3]:
#     print(f"i is {i}")
#     # print(combinations[i])
#     # Extract the validation accuracies
#     accs = re.findall('.*Val acc before epoch \d+: (\d+\.\d+)|.*Val acc at iteration \d+: (\d+\.\d+)', combinations[i])
#     accs = [acc[0] if acc[0] != '' else acc[1] for acc in accs]
#     # Convert to floats
#     accs = [float(acc) for acc in accs]
#     print(f"Length of accs: {len(accs)}")
#     # Apply moving average
#     accs_smooth = moving_average(np.array(accs))
#     print(np.mean(accs_smooth[470:489]))
#     if len(accs_smooth) >= 25:
#         if i != 18:
#             avg_last_25_epochs[i] = np.mean(accs_smooth[470:489])#-25: 470:489
#         else:
#             avg_last_25_epochs[i] = np.mean(accs_smooth[710:731])#-25: 470:489
#     # Add to the graph
#     plt.plot(accs_smooth, label=labels[i-1])  # Use the corresponding label

# # Add labels, title, and legend
# plt.title('Comparing Latent Weights (Smoothed, n=10)')
# # plt.title('Comparing Prototype Lengths (Smoothed, n=20)')
# plt.xlabel('Epoch')
# plt.ylabel('Validation Accuracy')
# plt.grid(True)
# plt.legend()
# # Show the graph
# plt.show()

# bars = []  # List to store the bars
# bar_labels = []  # List to store the bar labels

# for i in avg_last_25_epochs.keys():
#     bar = plt.bar(i, avg_last_25_epochs[i], color='C0')
#     bars.append(bar)  # Add the bar to the list
#     bar_labels.append(labels[i-1])  # Add the corresponding label to the list

# # plt.title('Comparing Prototype Lengths')
# plt.title('Latent Weight vs. Raw Input Weight')
# # plt.xlabel('Prototype Length')
# plt.xlabel('Latent Weight')
# plt.ylabel('End Validation Accuracy')
# plt.grid(True)
# plt.ylim(.9, 1)
# # Set the x-axis labels to the 'labels' list
# plt.xticks(ticks=range(1, len(labels)+1), labels=labels)
# plt.show()