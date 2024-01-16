import pandas as pd

# Define the csv files
csv_files = ['results_1-10.csv', 'results_64.csv', 'results_71.csv', 'results_12-31.csv', 'results.csv', 'model_based_search_results.csv']

# Initialize the maximum accuracy and corresponding datetime
max_accuracy = 0
max_datetime = None
max_file = None

# Loop over each csv file
for csv_file in csv_files:
    # Read the csv file
    df = pd.read_csv(csv_file)
    
    # Check if the maximum accuracy in this dataframe is greater than the current maximum accuracy
    if df['val_micro_accuracy'].max() > max_accuracy:
        # Update the maximum accuracy and corresponding datetime
        max_accuracy = df['val_micro_accuracy'].max()
        max_datetime = df.loc[df['val_micro_accuracy'] == max_accuracy, 'datetime'].values[0]
        max_file = csv_file

# Print the datetime of the row with the highest accuracy and the csv file it is in
print(f'The datetime of the row with the highest val_micro_accuracy is {max_datetime} in {max_file}.')

