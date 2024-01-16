import os
import shutil

# Define the directories
source_directory = 'old_best_saved_models'
destination_directory = 'saved_models'

# Check if the destination directory exists, if not, create it
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Loop over each file in the source directory
for file_name in os.listdir(source_directory):
    # Construct the full file paths
    source_file_path = os.path.join(source_directory, file_name)
    destination_file_path = os.path.join(destination_directory, file_name)
    
    # Move the file
    shutil.move(source_file_path, destination_file_path)

print(f'All files moved from {source_directory} to {destination_directory}.')

