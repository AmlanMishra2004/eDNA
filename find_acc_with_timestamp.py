import pandas as pd

# Define the csv files
csv_files = ['results_1-10.csv', 'results_64.csv', 'results_71.csv', 'results_12-31.csv', 'results.csv', 'model_based_search_results.csv']


# Define the datetime
datetime = '20240112_232905'

# Loop over each csv file
for csv_file in csv_files:
    # Read the csv file
    df = pd.read_csv(csv_file)
    
    # Check if the datetime exists in the dataframe
    if datetime in df['datetime'].values:
        # Get the accuracy of the model with the specified datetime
        accuracy = df.loc[df['datetime'] == datetime, 'val_micro_accuracy'].values[0]
        print(f'The accuracy of the model with datetime {datetime} in {csv_file} is {accuracy}.')
        break
else:
    print(f'The model with datetime {datetime} was not found in any of the csv files.')

