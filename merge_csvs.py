import pandas as pd

df = pd.read_csv('merged.csv')

# List of column indices that the warning mentioned
warning_cols = [22, 35, 36, 41, 43, 87]

df_no_duplicates = df

# Print data types and an example from each warned column
for col in warning_cols:
    # Convert index to column name
    col_name = df_no_duplicates.columns[col]
    print(f"Column: {col_name}")
    
    # Get unique data types in the column
    unique_dtypes = df_no_duplicates[col_name].apply(lambda x: type(x)).unique()
    
    for dtype in unique_dtypes:
        # Get an example value of this data type
        example_value = df_no_duplicates[col_name].apply(lambda x: x if type(x) == dtype else None).dropna().iloc[0]
        print(f"Data type: {dtype}")
        print(f"Example value: {example_value}")
    print()

