import pandas as pd

# Define the csv files
csv_files = ['results_1-10.csv', 'results_64.csv', 'results_71.csv', 'results_12-31.csv', 'results.csv', 'model_based_search_results.csv']

# Read the first csv file
df = pd.read_csv(csv_files[0])

# Get the columns of the first csv file
columns = df.columns

# Loop over the rest of the csv files
for csv_file in csv_files[1:]:
    # Read the csv file
    df_temp = pd.read_csv(csv_file)
    
    # Check if the columns are the same
    if not df_temp.columns.equals(columns):
        # Get the columns that are different
        different_columns = df_temp.columns.difference(columns).union(columns.difference(df_temp.columns))
        print(f'The columns of {csv_file} are not the same as the columns of {csv_files[0]}. Different columns: {different_columns.tolist()}')
        break
    # Append the dataframe to the main dataframe
    df = df.append(df_temp, ignore_index=True)
else:
    # Save the unified dataframe to a new csv file
    df.to_csv('unified_results.csv', index=False)
    print('All csv files have been unified into unified_results.csv.')

