import pandas as pd

# Merges rows with the same 'name' value, and show mean and std dev
def merge_rows_with_same_name(csv_file):
    df = pd.read_csv(csv_file)
    
    grouped = df.groupby('name').agg(['mean', 'std'])
    merged_df = grouped.reset_index()

    if ('test_micro_accuracy', 'mean') in merged_df.columns:
        merged_df = merged_df.sort_values(by=('test_micro_accuracy', 'mean'), ascending=False)
    
    return merged_df

if __name__ == "__main__":
    for trainnoise in [0,1,2]: # ,1,2
        for testnoise in [0,1,2]:
            csv_file = f"baseline_results_same_as_zurich_oversampled_t70_trainnoise-{trainnoise}_testnoise-{testnoise}_thresh-2.csv"
            merged_df = merge_rows_with_same_name(csv_file)
            merged_df.to_csv(f"merged_{csv_file}.csv", index=False)
    print("finished")