import os
import pandas as pd
# from tqdm import tqdm

# Define the directory and csv files
directory = 'saved_models'
#csv_files = ['results_12-31.csv']
csv_files = ['results_1-10.csv', 'results_64.csv', 'results_71.csv', 'results_12-31.csv', 'results.csv', 'model_based_search_results.csv']


# Calculate the total number of rows across all csv files
total_rows = sum([sum(1 for row in open(csv_file)) for csv_file in csv_files]) - len(csv_files)  # subtract header rows
curr_row = 0

# Initialize the progress bar
# pbar = tqdm(total=total_rows)

# Define the temporary directory
temp_dir = 'old_best_saved_models'

# Create the temporary directory if it doesn't exist
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Loop over each csv file
for csv_file in csv_files:
    # Read the csv file
    df = pd.read_csv(csv_file)
    
    # Filter rows where 'val_micro_accuracy' is less than 0.95
    filtered_df = df[df['val_micro_accuracy'] < 0.983]
    
    # Loop over each row in the filtered dataframe
    for index, row in filtered_df.iterrows():
        curr_row += 1
        if curr_row % 50 == 0:
            print(f"On row {curr_row} / {total_rows}")

        # Construct the file name
        file_name = 'best_model_' + row['datetime'] + '.pt'
        
        # Construct the full file path
        file_path = os.path.join(directory, file_name)
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Move the file to the temporary directory
            os.rename(file_path, os.path.join(temp_dir, file_name))
            # print(f'Moved file: {file_path} to {temp_dir}')
        
        # Update the progress bar
        # pbar.update(1)

# Close the progress bar
# pbar.close()

